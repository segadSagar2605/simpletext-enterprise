from dotenv import load_dotenv
load_dotenv()  # Must be first — loads GEMINI_API_KEY before any other import uses it

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
import time
import asyncio
from datetime import datetime
from google import genai
from google.genai import types
from .database import get_db_conn, init_db
from .services.indexer import background_content_indexing, collection, get_embeddings_batch, generate_summary
from .services.extraction import extract_text_from_file
from .utils.logger import log_event, PerformanceTimer
from .utils.performance_broadcaster import (
    register_broadcast_listener,
    unregister_broadcast_listener,
    SimplePerformanceFormatter,
    event_broadcaster_task
)
from contextlib import asynccontextmanager
from flashrank import Ranker, RerankRequest
import uuid
from .agent.graph import run_agent

# ============ GEMINI SETUP ============
# 1. Use the default client initialization that worked in your script
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# 2. Use the EXACT name found by your checkmodels.py script
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_GEN_MODEL = "gemini-2.5-flash-lite"
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


# ============ QUERY REWRITING (Option C) ============
# Detects ambiguous queries and rewrites them using condensed history.
# condensed_history: pre-joined summaries from DB — not raw chunks.
# Returns original query if self-contained or no history available.

def rewrite_query_if_needed(query: str, condensed_history: str) -> str:

    # If no history — nothing to rewrite against, return as-is
    if not condensed_history.strip():
        return query

    prompt = f"""You are a query rewriting assistant for an enterprise document search system.

Given this conversation context:
{condensed_history}

And this new question:
{query}

Your job:
1. Decide if the question is self-contained and unambiguous on its own.
2. If YES — return the original question exactly as-is.
3. If NO — rewrite it into a complete, self-contained question using the context above.
4. If the question is on a completely different topic from the context — return it as-is.

Rules:
- Return ONLY the final question. No explanation. No preamble.
- Never add information not present in the context or question.
- Keep the rewritten question concise and specific.

Final question:"""

    try:
        response = client.models.generate_content(
            model=GEMINI_GEN_MODEL,
            contents=prompt
        )
        rewritten = response.text.strip()

        # Safety check — if Gemini returns empty, fall back to original
        return rewritten if rewritten else query

    except Exception as e:
        # Never break the pipeline — fall back to original query on any error
        print(f"[Query Rewriting] Gemini error: {e}")
        return query

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


# ============ REQUEST MODELS ============

class ConversationTurn(BaseModel):
    role: str        # 'user' or 'assistant'
    content: str

class AskRequest(BaseModel):
    q: str
    doc_type: Optional[str] = None   # reserved for future filtering
    session_id: Optional[str] = None      # UUID for conversation thread
    conversation_history: List[ConversationTurn] = []  # previous turn


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


@app.get("/doc/{doc_id}")
def get_doc(doc_id: int):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, created_by, content_summary, doc_type, created_at, file_path, status
        FROM documents WHERE id = ?
    """, (doc_id,))
    r = cursor.fetchone()
    conn.close()
    if not r:
        return {}
    return {"id": r[0], "title": r[1], "author": r[2], "summary": r[3],
            "type": r[4], "date": r[5], "path": r[6], "status": r[7]}


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

    # Try full hybrid search (LIKE + FTS5)
    try:
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
    except Exception:
        # FTS5 failed (bad query syntax / cold index) — fall back to LIKE only
        cursor.execute("""
            SELECT id, title, created_by, content_summary, doc_type, created_at, file_path, status
            FROM documents
            WHERE title LIKE ? OR content_summary LIKE ?
        """, (search_query, search_query))

    rows = cursor.fetchall()
    conn.close()

    return [{
        "id": r[0],
        "title": r[1], "author": r[2], "summary": r[3],
        "type": r[4], "date": r[5], "path": r[6], "status": r[7]
    } for r in rows]


@app.post("/preview")
async def preview_document(file: UploadFile = File(...)):
    """
    Step 1 of two-step upload.
    Accepts a file, extracts text, generates AI summary.
    Returns summary to UI so user can review/edit before uploading.
    File is NOT saved permanently here — just read and discarded.
    """
    import tempfile

    # Write to a temp file so extract_text_from_file can read it
    suffix = os.path.splitext(file.filename)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        full_text = extract_text_from_file(tmp_path)
        if not full_text:
            return {"summary": "", "error": "Could not extract text from this file type."}

        # Use smart sampling — same logic as indexer
        from .services.indexer import recursive_splitter
        parent_blocks = recursive_splitter(full_text, max_size=1000)

        # Strategic sampling: first + middle + last chunks
        blocks = [parent_blocks[0]]
        if len(parent_blocks) > 2:
            mid = len(parent_blocks) // 3
            blocks += parent_blocks[mid: mid + 2]
        if len(parent_blocks) > 1:
            blocks.append(parent_blocks[-1])

        sample = "\n\n---\n\n".join(blocks)

        title_hint = os.path.splitext(file.filename)[0]
        summary = generate_summary(sample, title_hint)

        return {"summary": summary, "error": None}

    except Exception as e:
        return {"summary": "", "error": str(e)}
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/ask")
async def ask_neural_assistant(request: AskRequest):
    q = request.q
    session_id = request.session_id or str(uuid.uuid4())
    # SLIDING WINDOW — cap history to last 10 turns to protect LLM context window
    # Older turns remain in SQLite but are not sent to the model
    MAX_HISTORY_TURNS = 10
    history = request.conversation_history[-MAX_HISTORY_TURNS:]
    conn = get_db_conn()                                   
    cursor = conn.cursor()

    # 0. QUERY REWRITING (Option C)
    # Load summaries from DB for this session — not raw chunks
    # Join into condensed history and rewrite if query is ambiguous
    cursor.execute("""
        SELECT summary FROM conversation_memory
        WHERE session_id = ? AND role = 'assistant' AND summary IS NOT NULL
        ORDER BY created_at DESC
        LIMIT ?
    """, (session_id, MAX_HISTORY_TURNS))
    summary_rows = cursor.fetchall()
    condensed_history = " | ".join([row[0] for row in reversed(summary_rows)])
    q = rewrite_query_if_needed(q, condensed_history)



    # 1. NEURAL SEARCH — embed the query with Gemini (task_type: RETRIEVAL_QUERY)
    result = client.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=q,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = [result.embeddings[0].values]

    vector_results = collection.query(query_embeddings=query_vector, n_results=5)

    # 2. KEYWORD SEARCH (SQLite FTS5)

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
    

    # 5. LIGHTWEIGHT RERANKING
    if passages_for_reranking:
        rerank_request = RerankRequest(query=q, passages=passages_for_reranking)
        results = ranker.rerank(rerank_request)
        top_passages = [r['text'] for r in results[:3]]
        top_scores = [float(round(r['score'], 4)) for r in results[:3]]
    else:
        top_passages = []
        top_scores = []

    # 6. SAVE TO MEMORY (Option C — summarise at save time)
    # Generate a compact summary of this turn before saving.
    # This summary is used by the query rewriter next turn — not raw chunks.
    now = datetime.now().isoformat()
    assistant_content = "\n\n---\n\n".join(top_passages)

    # Generate turn summary — one Gemini call, invisible to user
    try:
        summary_prompt = f"""Summarise this Q&A turn in under 100 words.
                            Focus on key entities, decisions, and conclusions only.
                            Return ONLY the summary. No preamble.

                            Q: {q}
                            A: {assistant_content[:2000]}"""

        summary_response = client.models.generate_content(
            model=GEMINI_GEN_MODEL,
            contents=summary_prompt
        )
        turn_summary = summary_response.text.strip()
    except Exception as e:
        print(f"[Summary Generation] Gemini error: {e}")
        turn_summary = None

    # Save user turn — no summary needed for user questions
    cursor.execute(
        "INSERT INTO conversation_memory (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, "user", q, now)
    )
    # Save assistant turn WITH summary
    cursor.execute(
        "INSERT INTO conversation_memory (session_id, role, content, summary, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, "assistant", assistant_content, turn_summary, now)
    )
    conn.commit()
    conn.close()

    return {
        "query": q,
        "session_id": session_id,
        "retrieved_context": assistant_content,
        "rerank_scores": top_scores,
        "sources_found": len(passages_for_reranking),
        "history": [{"role": h.role, "content": h.content} for h in history]
    }

# ============ HISTORY ENDPOINT ============
@app.get("/history/{session_id}")
def get_conversation_history(session_id: str):
    """
    Retrieves all conversation turns for a given session_id.
    Used to restore context at the start of a new session.
    Returns turns in chronological order (oldest first).
    """
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, content, created_at
        FROM conversation_memory
        WHERE session_id = ?
        ORDER BY created_at ASC
    """, (session_id,))
    rows = cursor.fetchall()
    conn.close()

    return {
        "session_id": session_id,
        "turns": [
            {"role": r[0], "content": r[1], "created_at": r[2]}
            for r in rows
        ]
    }


# ============ AGENT ENDPOINT ============
@app.post("/agent")
async def run_agent_endpoint(request: AskRequest):
    """
    Agentic endpoint — routes query through LangGraph.
    Router decides: RAG only, web search only, or both in parallel.
    Returns synthesised final answer.
    """
    result = run_agent(
        query=request.q,
        session_id=request.session_id
    )
    return result


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
