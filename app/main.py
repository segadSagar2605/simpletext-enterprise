import shutil
import logging
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db, get_db_conn
from app.services.indexer import background_content_indexing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleText")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Verifying environment...")
    Path("uploads").mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info("Environment ready. SimpleText Enterprise online.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="SimpleText Enterprise", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form("Untitled"),
    created_by: str = Form("System"),
    content_summary: str = Form(""),
    doc_type: str = Form("PDF")
):
    file_path = Path("uploads") / file.filename
    
    try:
        # Save physical file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        conn = get_db_conn()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        cursor.execute('''
            INSERT INTO documents (title, created_by, content_summary, doc_type, file_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (title, created_by, content_summary, doc_type, str(file_path), timestamp))
        
        conn.commit()
        conn.close()

        background_tasks.add_task(background_content_indexing, title, content_summary, str(file_path))
        
        return {"status": "success", "message": f"'{title}' ingested successfully"}

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/get-docs")
def list_docs():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT title, created_by, content_summary, doc_type, created_at FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"title": r[0], "author": r[1], "summary": r[2], "type": r[3], "date": r[4]} for r in rows]

@app.get("/search")
def search_docs(q: str):
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT title FROM doc_search WHERE doc_search MATCH ?", (f"*{q}*",))
    results = cursor.fetchall()
    conn.close()
    return [{"title": r[0]} for r in results]