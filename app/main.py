from dotenv import load_dotenv
load_dotenv()  # Must be first — loads GEMINI_API_KEY before any other import uses it

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import time
import asyncio
from datetime import datetime
from google import genai
from google.genai import types
from .database import get_db_conn, init_db
from .services.indexer import background_content_indexing, collection, get_embeddings_batch
from .utils.logger import log_event, PerformanceTimer
from .utils.performance_broadcaster import (
    register_broadcast_listener,
    unregister_broadcast_listener,
    SimplePerformanceFormatter,
    event_broadcaster_task
)
from contextlib import asynccontextmanager
from flashrank import Ranker, RerankRequest

# ============ GEMINI SETUP ============
# 1. Use the default client initialization that worked in your script
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# 2. Use the EXACT name found by your checkmodels.py script
GEMINI_EMBED_MODEL = "gemini-embedding-001"

# ============ RERANKER ============
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")


# ============ WEBSOCKET CONNECTION MANAGER ============
class ConnectionManager:
    """Manages active WebSocket connections for pipeline status broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event_data: dict):
        """Send event to all connected clients."""
        message = SimplePerformanceFormatter.format_for_display(event_data)
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "message": message,
                    "data": event_data,
                    "type": event_data.get("type")
                })
            except Exception as e:
                print(f"WebSocket broadcast error: {e}")


manager = ConnectionManager()


# ============ BROADCAST HANDLER ============
async def websocket_broadcast_handler(event_data: dict):
    """Handler that bridges performance events to WebSocket connections."""
    await manager.broadcast(event_data)


# ============ APP LIFESPAN ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("[Startup] Initializing database schema...")
    init_db()

    # Register the WebSocket broadcaster
    register_broadcast_listener(websocket_broadcast_handler)

    # Start the event broadcaster background task
    broadcaster_task = asyncio.create_task(event_broadcaster_task())

    # Gemini is API-based — no local model to load, signal ready immediately
    await manager.broadcast({
        "type": "SYSTEM_READY",
        "message": "AI Engine Online (Gemini)",
        "timestamp": datetime.now().isoformat()
    })
    print("[System] AI Engine Online — using Gemini text-embedding-004")

    yield

    # --- SHUTDOWN ---
    broadcaster_task.cancel()
    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass

    unregister_broadcast_listener(websocket_broadcast_handler)
    print("[Shutdown] Cleaning up resources...")


# ============ APP INIT ============
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# ============ ENDPOINTS ============

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    created_by: str = Form("User"),
    content_summary: str = Form(""),
    doc_type: str = Form("PDF")
):
    log_event(None, "Upload Start", 0)
    upload_start = time.perf_counter()

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_duration = (time.perf_counter() - upload_start) * 1000
    log_event(None, "Upload Finish", upload_duration)

    with PerformanceTimer(None, "Btree Indexing Start", "Btree Indexing Finish"):
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (title, created_by, content_summary, doc_type, file_path, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 'Pending')
        """, (title, created_by, content_summary, doc_type, file_path, datetime.now().strftime("%Y-%m-%d %H:%M")))

        doc_id = cursor.lastrowid
        conn.commit()
    conn.close()

    background_tasks.add_task(background_content_indexing, doc_id, file_path, title)

    return {"status": "success", "doc_id": doc_id}


@app.get("/get-docs")
def list_docs():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, created_by, content_summary, doc_type, created_at, file_path, status
        FROM documents 
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    return [{
        "id": r[0],
        "title": r[1], "author": r[2], "summary": r[3],
        "type": r[4], "date": r[5], "path": r[6], "status": r[7]
    } for r in rows]


@app.get("/status")
def system_status():
    """Check if AI engine is ready. With Gemini, this is always instant."""
    return {"ready": True, "message": "AI Engine Online (Gemini text-embedding-004)"}


@app.get("/search")
def search_docs(q: str):
    """
    Hybrid Search: Checks Metadata (Title/Summary) AND Deep Content (FTS5).
    """
    conn = get_db_conn()
    cursor = conn.cursor()

    search_query = f"%{q}%"

    cursor.execute("""
        SELECT id, title, created_by, content_summary, doc_type, created_at, file_path, status
        FROM documents
        WHERE (title LIKE ? OR content_summary LIKE ?)
        OR id IN (
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
        "id": r[0],
        "title": r[1], "author": r[2], "summary": r[3],
        "type": r[4], "date": r[5], "path": r[6], "status": r[7]
    } for r in rows]


@app.post("/ask")
async def ask_neural_assistant(q: str = Form(...)):
    # 1. NEURAL SEARCH — embed the query with Gemini (task_type: RETRIEVAL_QUERY)
    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=q,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = [result.embeddings[0].values]

    vector_results = collection.query(query_embeddings=query_vector, n_results=5)

    # 2. KEYWORD SEARCH (SQLite FTS5)
    conn = get_db_conn()
    cursor = conn.cursor()

    clean_q = q.replace('"', ' ').strip()

    if not clean_q:
        keyword_ids = []
    else:
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
        row = cursor.fetchone()
        if row:
            passages_for_reranking.append({"id": p_id, "text": row[0]})
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


# ============ WEBSOCKET ENDPOINT ============
@app.websocket("/ws/pipeline_status")
async def websocket_pipeline_status(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pipeline status updates.
    Clients connect here to receive live performance events.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
