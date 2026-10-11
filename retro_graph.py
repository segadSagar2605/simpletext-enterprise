"""
retro_graph.py — One-time backfill script for GraphRAG Phase 3.

Run this ONCE from the project root after deploying graph_builder.py
to extract entities + relationships from documents that were indexed
before Phase 3 was introduced.

Usage:
    python retro_graph.py              # process all unprocessed docs
    python retro_graph.py --dry-run    # preview what would be processed
    python retro_graph.py --doc-id 7   # process one specific document

Safe to re-run — already processed parent chunks are skipped automatically.
Rate-limit aware — adds a small delay between documents to respect Gemini limits.

Progress is printed to console. Run from project root:
    cd C:\\Users\\foodiesagar2605\\Desktop\\simpletext_enterprise
    python retro_graph.py
"""

import sys
import os
import time
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
# Adds project root to sys.path so app.* imports work without installation
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load .env if present (picks up GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    print("[Retro] Loaded .env")
except ImportError:
    pass  # dotenv not installed — rely on environment variable being set

from app.database import get_db_conn
from app.services.graph_builder import build_graph_for_document

# ── Config ────────────────────────────────────────────────────────────────────
# Delay between documents (seconds) — prevents Gemini rate limit errors.
# Increase to 3.0 if you hit 429 errors on large document sets.
INTER_DOC_DELAY = 1.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_all_documents(conn, doc_id_filter=None):
    """
    Returns list of (doc_id, title) for all Ready documents.
    Optionally filtered to a single doc_id.
    """
    if doc_id_filter:
        rows = conn.execute(
            "SELECT id, title FROM documents WHERE id = ? AND status = 'Ready'",
            (doc_id_filter,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, title FROM documents WHERE status = 'Ready' ORDER BY id"
        ).fetchall()
    return rows


def get_unprocessed_parents(conn, doc_id):
    """
    Returns (parent_ids, parent_texts) for parent chunks that have NOT
    yet been processed through graph extraction.

    A chunk is considered processed if at least one entity row exists
    for that parent_id — safe skip, no duplicate extraction.
    """
    all_parents = conn.execute(
        "SELECT id, content FROM parents WHERE doc_id = ? ORDER BY id",
        (doc_id,)
    ).fetchall()

    if not all_parents:
        return [], []

    # Find which parent_ids already have entities extracted
    processed = set(
        row[0] for row in conn.execute(
            "SELECT DISTINCT parent_id FROM entities WHERE doc_id = ?",
            (doc_id,)
        ).fetchall()
    )

    unprocessed = [(pid, text) for pid, text in all_parents if pid not in processed]
    return (
        [row[0] for row in unprocessed],
        [row[1] for row in unprocessed],
    )


def print_separator():
    print("-" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(dry_run=False, doc_id_filter=None):
    print()
    print("=" * 60)
    print("  SimpleText — GraphRAG Retro Backfill")
    print("=" * 60)

    if dry_run:
        print("  MODE: DRY RUN — no data will be written")
    if doc_id_filter:
        print(f"  FILTER: doc_id = {doc_id_filter}")
    print()

    conn = get_db_conn()
    docs = get_all_documents(conn, doc_id_filter)

    if not docs:
        print("[Retro] No Ready documents found. Nothing to process.")
        conn.close()
        return

    print(f"[Retro] Found {len(docs)} document(s) to check.\n")

    total_entities = 0
    total_edges    = 0
    total_skipped  = 0
    total_processed_docs = 0

    for doc_id, title in docs:
        print_separator()
        print(f"[Retro] Document {doc_id}: '{title}'")

        parent_ids, parent_texts = get_unprocessed_parents(conn, doc_id)

        if not parent_ids:
            print(f"  → Already fully processed. Skipping.")
            total_skipped += 1
            continue

        print(f"  → {len(parent_ids)} unprocessed chunk(s) found.")

        if dry_run:
            print(f"  → DRY RUN: would extract graph from {len(parent_ids)} chunks.")
            continue

        # ── Run graph extraction ───────────────────────────────────────────
        result = build_graph_for_document(
            doc_id=doc_id,
            parent_ids=parent_ids,
            parent_texts=parent_texts,
            doc_title=title,
            conn=None,   # retro path — graph_builder opens its own conn + commits
        )

        total_entities       += result["entities_added"]
        total_edges          += result["edges_added"]
        total_processed_docs += 1

        print(
            f"  → Done: {result['chunks_processed']} chunks | "
            f"{result['entities_added']} entities | "
            f"{result['edges_added']} edges"
        )

        # Rate limit buffer between documents
        if len(docs) > 1:
            print(f"  → Waiting {INTER_DOC_DELAY}s before next document...")
            time.sleep(INTER_DOC_DELAY)

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Retro Backfill Complete")
    print("=" * 60)
    print(f"  Documents processed : {total_processed_docs}")
    print(f"  Documents skipped   : {total_skipped} (already done)")
    print(f"  Total entities added: {total_entities}")
    print(f"  Total edges added   : {total_edges}")
    print("=" * 60)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill GraphRAG entities + edges for existing documents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be processed without writing anything.",
    )
    parser.add_argument(
        "--doc-id",
        type=int,
        default=None,
        help="Process a single document by id (useful for testing).",
    )
    args = parser.parse_args()

    run(dry_run=args.dry_run, doc_id_filter=args.doc_id)
