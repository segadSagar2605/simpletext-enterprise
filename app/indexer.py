import time
import os
from google import genai
from google.genai import types
from .extraction import extract_text_from_file
from ..database import get_db_conn
from ..utils.logger import log_event, PerformanceTimer
from ..utils.performance_broadcaster import (
    broadcast_event_sync,
    PerformanceTimerWithBroadcastSync
)
import chromadb

# ============ GEMINI EMBEDDING SETUP ============
# API key is loaded from .env via load_dotenv() in main.py
# 1. Use the default client initialization that worked in your script
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# 2. Use the EXACT name found by your checkmodels.py script
GEMINI_EMBED_MODEL = "gemini-embedding-001"

# ChromaDB client and collection
# Use absolute path anchored to project root (same pattern as database.py)
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SERVICES_DIR))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="enterprise_docs")


def get_embeddings_batch(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Batch embed using Gemini text-embedding-004.
    - Automatically handles Gemini's 100-text-per-call limit.
    - task_type: 'RETRIEVAL_DOCUMENT' for indexing, 'RETRIEVAL_QUERY' for search.
    """
    BATCH_SIZE = 100
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        all_embeddings.extend([e.values for e in result.embeddings])

    return all_embeddings


def recursive_splitter(text, max_size=1000):
    """
    The 'Smart Scissors': Splitting text by paragraphs, then sentences.
    Ensures thoughts like headers and rows stay together.
    """
    if len(text) <= max_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) <= max_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def background_content_indexing(doc_id: int, file_path: str, doc_title: str = None):
    """
    Parent-Child Indexing Engine with BATCH PROCESSING + Gemini Embeddings.
    Optimizations:
    - Gemini text-embedding-004: higher quality 768-dim vectors, no local model RAM
    - Batch Embedding: collect ALL chunks, one API call per 100 chunks
    - Batch DB Commits: ONE final commit at the end (not per chunk)
    - Status Transitions: Pending → Indexing → Ready
    """
    broadcast_event_sync(doc_id, "Content Indexing Start", 0, doc_title)

    # Update status to 'Indexing'
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("UPDATE documents SET status = 'Indexing' WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error] Failed to set Indexing status: {e}")

    full_text = extract_text_from_file(file_path)
    if not full_text:
        print(f"[Error] No text found in {file_path}")
        broadcast_event_sync(doc_id, "Content Indexing Failed", 0, doc_title)
        try:
            conn = get_db_conn()
            cursor = conn.cursor()
            cursor.execute("UPDATE documents SET status = 'Failed' WHERE id = ?", (doc_id,))
            conn.commit()
            conn.close()
        except:
            pass
        return

    conn = get_db_conn()
    cursor = conn.cursor()

    # Chunking phase
    parent_blocks = recursive_splitter(full_text, max_size=1000)
    print(f"[Chunking] Created {len(parent_blocks)} parent blocks")

    # BATCH PROCESSING: Collect all data first
    all_parent_ids = []
    all_parent_texts = []
    all_child_chunks = []       # Flattened list of ALL chunks from ALL parents
    chunk_metadata = []         # Track which parent each chunk belongs to

    for idx, p_text in enumerate(parent_blocks):
        parent_id = f"{doc_id}_p{idx:03d}"
        all_parent_ids.append(parent_id)
        all_parent_texts.append(p_text)

        # Create children (~300 chars) for vectors
        child_chunks = [p_text[i:i+300] for i in range(0, len(p_text), 250)]
        for j, chunk in enumerate(child_chunks):
            all_child_chunks.append(chunk)
            chunk_metadata.append({
                "parent_idx": idx,
                "chunk_idx": j,
                "parent_id": parent_id
            })

    # INSERT PARENTS & FTS5
    print(f"[FTS] Indexing {len(all_parent_ids)} parents for full-text search")
    for parent_id, p_text in zip(all_parent_ids, all_parent_texts):
        cursor.execute(
            "INSERT INTO parents (id, doc_id, content) VALUES (?, ?, ?)",
            (parent_id, doc_id, p_text)
        )
        cursor.execute(
            "INSERT INTO doc_search (parent_id, content) VALUES (?, ?)",
            (parent_id, p_text)
        )

    # BATCH EMBEDDING: Gemini API call(s) — no local model, no RAM overhead
    print(f"[Embedding] Starting Gemini batch encoding of {len(all_child_chunks)} chunks...")
    embedding_start = time.perf_counter()

    if all_child_chunks:
        all_embeddings = get_embeddings_batch(all_child_chunks, task_type="RETRIEVAL_DOCUMENT")
        embedding_duration = (time.perf_counter() - embedding_start) * 1000
        print(f"[Gemini] Encoding complete in {embedding_duration:.0f}ms.")
    else:
        all_embeddings = []
        embedding_duration = 0

    # Add all to ChromaDB in one batch
    chroma_batch_ids = []
    chroma_batch_embeddings = []
    chroma_batch_documents = []
    chroma_batch_metadatas = []

    for embedding_idx, (embedding, meta) in enumerate(zip(all_embeddings, chunk_metadata)):
        chunk_idx = meta["chunk_idx"]
        parent_id = meta["parent_id"]

        chroma_batch_ids.append(f"{parent_id}_c{chunk_idx}")
        chroma_batch_embeddings.append(embedding)
        chroma_batch_documents.append(all_child_chunks[embedding_idx])
        chroma_batch_metadatas.append({
            "parent_id": parent_id,
            "doc_id": doc_id
        })

    if chroma_batch_ids:
        collection.add(
            ids=chroma_batch_ids,
            embeddings=chroma_batch_embeddings,
            documents=chroma_batch_documents,
            metadatas=chroma_batch_metadatas
        )

    # SINGLE FINAL COMMIT
    try:
        conn.commit()

        cursor.execute("UPDATE documents SET status = 'Ready' WHERE id = ?", (doc_id,))
        conn.commit()

        broadcast_event_sync(doc_id, "Document Ready", 0, doc_title)

        print(f"[Success] Document {doc_id} indexed: {len(parent_blocks)} parents, {len(all_child_chunks)} chunks.")
    except Exception as e:
        print(f"[Error] Final commit failed: {e}")
        broadcast_event_sync(doc_id, "Content Indexing Failed", 0, doc_title)
        try:
            cursor.execute("UPDATE documents SET status = 'Failed' WHERE id = ?", (doc_id,))
            conn.commit()
        except:
            pass
    finally:
        conn.close()
