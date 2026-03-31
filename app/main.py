from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
from .database import get_db_conn
from .services.indexer import background_content_indexing
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load once on startup to save RAM
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

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
    # Save physical file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 1: Execute B-Tree indexing for Metadata
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (title, created_by, content_summary, doc_type, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (title, created_by, content_summary, doc_type, file_path, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

    # Step 2: Trigger FTS5 indexing for Content in the background
    background_tasks.add_task(background_content_indexing, title, content_summary, file_path)
    
    return {"status": "success"}

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
    Finds documents using FTS5 for deep-content matching.
    Uses a subquery to avoid 'MATCH context' operational errors in SQLite.
    """
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # --- PART 4: FTS5 SUBQUERY ---
    # We find matching titles in the Virtual Table first, then fetch their metadata.
    # This is the standard 'Enterprise' way to join FTS5 with B-Tree tables.
    cursor.execute("""
        SELECT title, created_by, content_summary, doc_type, created_at, file_path
        FROM documents
        WHERE title IN (
            SELECT title 
            FROM doc_search 
            WHERE doc_search MATCH ?
        )
    """, (q,))
    
    rows = cursor.fetchall()
    conn.close()
    
    # Return the results to the frontend
    return [{
        "title": r[0], "author": r[1], "summary": r[2], 
        "type": r[3], "date": r[4], "path": r[5]
    } for r in rows]

@app.post("/ask")
async def ask_neural_assistant(q: str = Form(...)):
    """
    NEURAL SEARCH: This handles the 'Brain' work.
    It doesn't look for keywords; it looks for answers.
    """
    query_vector = model.encode([q]).tolist()
    
    results = collection.query(
        query_embeddings=query_vector,
        n_results=3
    )
    
    # Return chunks of text as 'answers'
    return {
        "answer_context": results['documents'][0],
        "sources": [m['file_name'] for m in results['metadatas'][0]]
    }