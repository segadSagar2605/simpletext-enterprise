from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
from .database import get_db_conn
from .services.indexer import background_content_indexing
import chromadb
from sentence_transformers import SentenceTransformer
from .services.indexer import model, collection # Import your existing AI models
from contextlib import asynccontextmanager
from .database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    # This runs BEFORE the server starts listening for requests
    print("Startup: Checking and initializing database schema...")
    init_db() 
    yield
    # --- SHUTDOWN LOGIC ---
    # You can add cleanup code here (e.g., closing DB connections) if needed
    print("Shutdown: Cleaning up resources...")

# Initialize the app with the lifespan manager
app = FastAPI(lifespan=lifespan)



#  INITIALIZE THE RERANKER
from flashrank import Ranker, RerankRequest
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")

#app = FastAPI()

#model = SentenceTransformer('all-MiniLM-L6-v2')
#chroma_client = chromadb.PersistentClient(path="./chroma_db")
#collection = chroma_client.get_or_create_collection(name="enterprise_docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    created_by: str = Form("User"),
    content_summary: str = Form(""),
    doc_type: str = Form("PDF")
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (title, created_by, content_summary, doc_type, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (title, created_by, content_summary, doc_type, file_path, datetime.now().strftime("%Y-%m-%d %H:%M")))
    
    # --- THE CRITICAL FIX ---
    doc_id = cursor.lastrowid  # Get the ID of the file we just saved
    conn.commit()
    conn.close()

    # Pass the doc_id, NOT the title/summary
    background_tasks.add_task(background_content_indexing, doc_id, file_path)
    
    return {"status": "success", "doc_id": doc_id}

@app.get("/get-docs")
def list_docs():
    # Standard retrieve for the dashboard using B-Tree index on created_at
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, created_by, content_summary, doc_type, created_at, file_path 
        FROM documents 
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "title": r[0], "author": r[1], "summary": r[2], 
        "type": r[3], "date": r[4], "path": r[5]
    } for r in rows]


@app.get("/search")
def search_docs(q: str):
    """
    Hybrid Search: Checks Metadata (Title/Summary) AND Deep Content (FTS5).
    """
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # We use LIKE for metadata and MATCH for the deep content
    search_query = f"%{q}%"
    
    cursor.execute("""
        SELECT title, created_by, content_summary, doc_type, created_at, file_path
        FROM documents
        WHERE (title LIKE ? OR content_summary LIKE ?) -- Step 1: Check Metadata
        OR id IN (                                     -- Step 2: Check Deep Content
            SELECT DISTINCT doc_id 
            FROM parents 
            WHERE id IN (
                SELECT parent_id 
                FROM doc_search 
                WHERE content MATCH ?
            )
        )
    """, (search_query, search_query, q))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{
        "title": r[0], "author": r[1], "summary": r[2], 
        "type": r[3], "date": r[4], "path": r[5]
    } for r in rows]

@app.post("/ask")
async def ask_neural_assistant(q: str = Form(...)):
    # 1. NEURAL SEARCH (ChromaDB)
    query_vector = model.encode([q]).tolist()
    vector_results = collection.query(query_embeddings=query_vector, n_results=5)
    
    # 2. KEYWORD SEARCH (SQLite FTS5)
# 2. KEYWORD SEARCH (SQLite FTS5)
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # --- THE ROBUST FIX ---
    # 1. Clean the query: Remove any double quotes the user might have typed
    clean_q = q.replace('"', ' ').strip()
    
    # 2. Handle Empty Queries: If q is empty, don't even run the FTS5 search
    if not clean_q:
        keyword_ids = []
    else:
        # 3. Wrapping in Quotes: This tells FTS5 to treat the whole string as one phrase.
        # This prevents "syntax error" when there are spaces or dashes.
        # Also, we match against the table name 'doc_search' for better reliability.
        fts_query = f'"{clean_q}"' 
        try:
            cursor.execute("SELECT parent_id FROM doc_search WHERE doc_search MATCH ? LIMIT 5", (fts_query,))
            keyword_ids = [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"FTS5 Search Error: {e}")
            keyword_ids = []

    # 3. IDENTITY FUSION (Deduplication)
    candidate_parent_ids = set(keyword_ids) 
    for metadata in vector_results['metadatas'][0]:
        candidate_parent_ids.add(metadata['parent_id'])

    # 4. PARENT RETRIEVAL (Fetching Full Context)
    passages_for_reranking = []
    for p_id in candidate_parent_ids:
        cursor.execute("SELECT content FROM parents WHERE id = ?", (p_id,))
        result = cursor.fetchone()
        if result:
            passages_for_reranking.append({"id": p_id, "text": result[0]})
    conn.close()

    # 5. LIGHTWEIGHT RERANKING
    if passages_for_reranking:
        rerank_request = RerankRequest(query=q, passages=passages_for_reranking)
        results = ranker.rerank(rerank_request)
        top_passages = [r['text'] for r in results[:3]]
    else:
        top_passages = []

    return {
        "query": q,
        "retrieved_context": "\n\n---\n\n".join(top_passages),
        "sources_found": len(passages_for_reranking)
    }