import time
import os
from google import genai
from google.genai import types
from .extraction import extract_text_from_file
from ..database import get_db_conn
from ..utils.logger import log_event, PerformanceTimer
from ..utils.performance_broadcaster import (
    broadcast_event_sync,
    PerformanceTimerWithBroadcastSync,
)
import chromadb

# ── Gemini setup ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_EMBED_MODEL   = "gemini-embedding-001"
GEMINI_SUMMARY_MODEL = "gemini-2.0-flash-lite"

# ── ChromaDB setup ────────────────────────────────────────────────────────────
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SERVICES_DIR))
CHROMA_PATH   = os.path.join(_PROJECT_ROOT, "chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_or_create_collection(name="enterprise_docs")

EMBED_BATCH_SIZE = 100


# ── Summary generation ────────────────────────────────────────────────────────

def generate_summary(text_sample: str, doc_title: str = None) -> str:
    """
    Generate a rich 5-sentence summary using Gemini Flash.

    Covers:
        1. What the document is about
        2. Its key purpose
        3. Important context (project, dates, people)
        4. Key numbers, metrics, or dates mentioned
        5. Who it is relevant for / who authored it

    Used in two places:
        - indexer.py  : after extraction, saved to DB automatically
        - main.py     : /preview endpoint, returned to UI before upload

    Falls back to empty string on any error — indexing never fails
    because of summary generation.
    """
    try:
        title_hint = f'Document title: "{doc_title}"\n\n' if doc_title else ""

        prompt = (
            f"{title_hint}"
            f"Read the following document excerpt and write exactly 5 sentences "
            f"summarising the CONTENT of this document. "
            f"Focus only on what the document says — the ideas, decisions, "
            f"numbers, risks, findings, or conclusions it contains. "
            f"Do not mention the document type, author, or file format. "
            f"Be specific and factual. Use only information from the text.\n\n"
            f"Excerpt:\n{text_sample}"
        )

        response = client.models.generate_content(
            model=GEMINI_SUMMARY_MODEL,
            contents=prompt,
        )

        summary = response.text.strip()
        print(f"[Summary] Generated for '{doc_title}': {summary[:100]}...")
        return summary

    except Exception as e:
        print(f"[Summary] Failed for '{doc_title}': {e}")
        return ""


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embeddings_batch(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Batch-embed via Gemini, respecting the 100-text-per-call limit.
    Returns embeddings in the same order as input texts.
    """
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch  = texts[i : i + EMBED_BATCH_SIZE]
        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        all_embeddings.extend([e.values for e in result.embeddings])
    return all_embeddings


# ── Chunking ──────────────────────────────────────────────────────────────────

def recursive_splitter(text: str, max_size: int = 1000) -> list[str]:
    """
    Split by paragraphs first, then sentences.
    Keeps semantic units (headers, table rows) together.
    """
    if len(text) <= max_size:
        return [text]

    chunks, current = [], ""
    for p in text.split("\n\n"):
        if len(current) + len(p) <= max_size:
            current += p + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks


# ── Main indexing pipeline ────────────────────────────────────────────────────

def background_content_indexing(doc_id: int, file_path: str, doc_title: str = None):
    """
    Full parent-child indexing pipeline with granular performance telemetry.

    Phases:
        1. Text extraction
        2. Auto-summary generation (Gemini Flash) — strategic chunk sampling
        3. Chunking (parent-child)
        4. FTS / parent DB inserts
        5. Batch embedding (Gemini)
        6. ChromaDB upsert
        7. Final DB commit

    CSV events emitted:
        Content Indexing Start
        Text Extraction Start / Finish      value=ms
        Summary Generation Finish           value=ms   meta={summary_length, ai_generated}
        Chunking Finish                     value=ms   meta={parent_blocks, total_child_chunks, avg_chunk_chars}
        FTS Indexing Start / Finish         value=ms   meta={parents_indexed}
        Embedding Batch Start               value=batch_num  unit=count
        Embedding Batch Finish              value=ms   meta={batch, chunks_embedded, throughput_chunks_per_sec}
        ChromaDB Upsert Finish              value=ms   meta={vectors_added}
        DB Commit Finish                    value=ms
        Document Ready                      value=total_pipeline_ms
        Content Indexing Failed
    """
    pipeline_start = time.perf_counter()
    broadcast_event_sync(doc_id, "Content Indexing Start", 0, doc_title)

    # ── Mark Indexing in DB ───────────────────────────────────────────────────
    try:
        conn = get_db_conn()
        conn.execute("UPDATE documents SET status = 'Indexing' WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Error] Failed to set Indexing status: {e}")

    # ── Phase 1: Text extraction ──────────────────────────────────────────────
    with PerformanceTimer(doc_id, "Text Extraction Start", "Text Extraction Finish",
                          doc_title=doc_title):
        full_text = extract_text_from_file(file_path)

    if not full_text:
        print(f"[Error] No text found in {file_path}")
        _fail(doc_id, doc_title)
        return

    # ── Phase 2: Auto-summary generation ─────────────────────────────────────
    # Only generates if no human-provided summary exists in DB
    # Strategic sampling: first + middle + last chunks for full doc coverage
    summary_start = time.perf_counter()
    existing_summary = _get_existing_summary(doc_id)

    if not existing_summary:
        # Build chunks for sampling
        parent_blocks_for_summary = recursive_splitter(full_text, max_size=1000)

        # Strategic sample — beginning, middle, end
        sample_blocks = [parent_blocks_for_summary[0]]
        if len(parent_blocks_for_summary) > 2:
            mid = len(parent_blocks_for_summary) // 3
            sample_blocks += parent_blocks_for_summary[mid: mid + 2]
        if len(parent_blocks_for_summary) > 1:
            sample_blocks.append(parent_blocks_for_summary[-1])

        sample = "\n\n---\n\n".join(sample_blocks)
        summary = generate_summary(sample, doc_title)
        ai_generated = True
    else:
        # Human provided a summary at upload time — respect it
        summary = existing_summary
        ai_generated = False
        print(f"[Summary] Using human-provided summary for '{doc_title}'")

    summary_ms = (time.perf_counter() - summary_start) * 1000

    if summary:
        try:
            conn = get_db_conn()
            conn.execute(
                "UPDATE documents SET content_summary = ? WHERE id = ?",
                (summary, doc_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Summary] Failed to save: {e}")

    log_event(
        doc_id, "Summary Generation Finish",
        doc_title=doc_title,
        value=summary_ms, unit="ms",
        meta={
            "summary_length": len(summary),
            "ai_generated":   ai_generated,
        },
    )

    # ── Phase 3: Chunking ─────────────────────────────────────────────────────
    chunk_start   = time.perf_counter()
    parent_blocks = recursive_splitter(full_text, max_size=1000)

    all_parent_ids, all_parent_texts = [], []
    all_child_chunks: list[str] = []
    chunk_metadata:   list[dict] = []

    for idx, p_text in enumerate(parent_blocks):
        parent_id = f"{doc_id}_p{idx:03d}"
        all_parent_ids.append(parent_id)
        all_parent_texts.append(p_text)

        child_chunks = [p_text[i : i + 300] for i in range(0, len(p_text), 250)]
        for j, chunk in enumerate(child_chunks):
            all_child_chunks.append(chunk)
            chunk_metadata.append({"parent_idx": idx, "chunk_idx": j, "parent_id": parent_id})

    chunk_elapsed = (time.perf_counter() - chunk_start) * 1000
    avg_chars = int(sum(len(c) for c in all_child_chunks) / len(all_child_chunks)) if all_child_chunks else 0

    log_event(
        doc_id, "Chunking Finish",
        doc_title=doc_title,
        value=chunk_elapsed, unit="ms",
        meta={
            "parent_blocks":      len(parent_blocks),
            "total_child_chunks": len(all_child_chunks),
            "avg_chunk_chars":    avg_chars,
        },
    )
    print(f"[Chunking] {len(parent_blocks)} parents / {len(all_child_chunks)} chunks ({chunk_elapsed:.0f}ms)")

    # ── Phase 4: FTS / parent DB inserts ─────────────────────────────────────
    conn   = get_db_conn()
    cursor = conn.cursor()

    with PerformanceTimer(doc_id, "FTS Indexing Start", "FTS Indexing Finish",
                          doc_title=doc_title,
                          meta={"parents_indexed": len(all_parent_ids)}):
        for parent_id, p_text in zip(all_parent_ids, all_parent_texts):
            cursor.execute(
                "INSERT INTO parents (id, doc_id, content) VALUES (?, ?, ?)",
                (parent_id, doc_id, p_text),
            )
            cursor.execute(
                "INSERT INTO doc_search (parent_id, content) VALUES (?, ?)",
                (parent_id, p_text),
            )

    # ── Phase 5: Batch embedding ──────────────────────────────────────────────
    all_embeddings: list = []
    total_embedding_ms = 0.0
    num_batches = 0

    if all_child_chunks:
        num_batches = -(-len(all_child_chunks) // EMBED_BATCH_SIZE)

        for batch_num in range(num_batches):
            batch_start_idx = batch_num * EMBED_BATCH_SIZE
            batch            = all_child_chunks[batch_start_idx : batch_start_idx + EMBED_BATCH_SIZE]
            chunks_in_batch  = len(batch)

            log_event(
                doc_id, "Embedding Batch Start",
                doc_title=doc_title,
                value=batch_num + 1, unit="count",
                meta={
                    "batch":             batch_num + 1,
                    "chunks_in_batch":   chunks_in_batch,
                    "batch_size_limit":  EMBED_BATCH_SIZE,
                    "cumulative_chunks": batch_start_idx,
                },
            )

            t0 = time.perf_counter()
            result = client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            batch_embeddings = [e.values for e in result.embeddings]
            all_embeddings.extend(batch_embeddings)

            batch_ms   = (time.perf_counter() - t0) * 1000
            total_embedding_ms += batch_ms
            throughput = round(chunks_in_batch / (batch_ms / 1000), 1) if batch_ms > 0 else 0

            log_event(
                doc_id, "Embedding Batch Finish",
                doc_title=doc_title,
                value=batch_ms, unit="ms",
                meta={
                    "batch":                     batch_num + 1,
                    "chunks_embedded":           chunks_in_batch,
                    "throughput_chunks_per_sec": throughput,
                },
            )
            print(f"[Embedding] Batch {batch_num+1}/{num_batches} — {chunks_in_batch} chunks in {batch_ms:.0f}ms ({throughput} chunks/s)")

    # ── Phase 6: ChromaDB upsert ──────────────────────────────────────────────
    chroma_ids, chroma_embeddings, chroma_docs, chroma_metas = [], [], [], []
    for emb_idx, (embedding, meta) in enumerate(zip(all_embeddings, chunk_metadata)):
        chroma_ids.append(f"{meta['parent_id']}_c{meta['chunk_idx']}")
        chroma_embeddings.append(embedding)
        chroma_docs.append(all_child_chunks[emb_idx])
        chroma_metas.append({"parent_id": meta["parent_id"], "doc_id": doc_id})

    if chroma_ids:
        t0 = time.perf_counter()
        collection.add(
            ids=chroma_ids,
            embeddings=chroma_embeddings,
            documents=chroma_docs,
            metadatas=chroma_metas,
        )
        chroma_ms = (time.perf_counter() - t0) * 1000
        log_event(
            doc_id, "ChromaDB Upsert Finish",
            doc_title=doc_title,
            value=chroma_ms, unit="ms",
            meta={"vectors_added": len(chroma_ids)},
        )
        print(f"[ChromaDB] Upserted {len(chroma_ids)} vectors in {chroma_ms:.0f}ms")

    # ── Phase 7: Final DB commit ──────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        conn.commit()
        cursor.execute("UPDATE documents SET status = 'Ready' WHERE id = ?", (doc_id,))
        conn.commit()
        db_commit_ms = (time.perf_counter() - t0) * 1000

        log_event(
            doc_id, "DB Commit Finish",
            doc_title=doc_title,
            value=db_commit_ms, unit="ms",
        )

        total_pipeline_ms   = (time.perf_counter() - pipeline_start) * 1000
        avg_embed_per_chunk = round(total_embedding_ms / len(all_child_chunks), 3) if all_child_chunks else 0

        broadcast_event_sync(doc_id, "Document Ready", total_pipeline_ms, doc_title)
        log_event(
            doc_id, "Document Ready",
            doc_title=doc_title,
            value=total_pipeline_ms, unit="ms",
            meta={
                "parent_blocks":              len(parent_blocks),
                "child_chunks":               len(all_child_chunks),
                "embedding_batches":          num_batches,
                "total_embedding_ms":         round(total_embedding_ms, 1),
                "avg_embedding_ms_per_chunk": avg_embed_per_chunk,
            },
        )

        print(
            f"[Done] doc_id={doc_id}  parents={len(parent_blocks)}  "
            f"chunks={len(all_child_chunks)}  batches={num_batches}  "
            f"total={total_pipeline_ms:.0f}ms"
        )

    except Exception as e:
        print(f"[Error] Final commit failed: {e}")
        _fail(doc_id, doc_title)
    finally:
        conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_existing_summary(doc_id: int) -> str:
    """
    Returns human-provided summary if user typed one during upload.
    Returns empty string if blank — triggers AI generation.
    """
    try:
        conn = get_db_conn()
        row  = conn.execute(
            "SELECT content_summary FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        conn.close()
        return (row[0] or "").strip() if row else ""
    except Exception:
        return ""


def _fail(doc_id: int, doc_title: str = None):
    """Mark document as Failed in DB and broadcast."""
    broadcast_event_sync(doc_id, "Content Indexing Failed", 0, doc_title)
    try:
        conn = get_db_conn()
        conn.execute("UPDATE documents SET status = 'Failed' WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()
    except Exception:
        pass
