"""
graph_builder.py — Phase 3 GraphRAG entity + relationship extraction.

Runs at index time (called from indexer.py after FTS inserts) and
at retro time (called from retro_graph.py for existing documents).

Responsibilities:
    - Ask Gemini to extract entities + relationships from each parent chunk
    - Normalise entity names (lowercase, strip punctuation)
    - Upsert entities into the entities table (no duplicates per doc)
    - Upsert edges into the edges table
    - Emit telemetry events consistent with the rest of the pipeline
    - Never crash the indexing pipeline — all failures are logged and skipped

Tables written:
    entities (id, canonical_name, entity_type, doc_id, parent_id, created_at)
    edges    (id, source_id, target_id, relationship, weight, doc_id, created_at)
"""

import os
import json
import time
import re
from datetime import datetime, timezone

from google import genai
from ..database import get_db_conn
from ..utils.logger import log_event

# ── Gemini setup ──────────────────────────────────────────────────────────────
# Reuses the same client + model as indexer.py — no new dependencies
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_GEN_MODEL = "gemini-2.5-flash-lite"

# ── Extraction prompt ─────────────────────────────────────────────────────────
# Instructs Gemini to return strict JSON — no preamble, no markdown fences.
# entity_type is loosely guided but Gemini can add types not in the list.
# relationship is completely free-text — stored exactly as returned.
# weight reflects Gemini's confidence (0.0–1.0) that the relationship is real.

EXTRACTION_PROMPT = """\
You are an expert knowledge graph builder. Read the text below and extract:
1. Named entities — people, systems, teams, processes, milestones, or concepts
2. Directed relationships between those entities

Rules:
- canonical_name: lowercase, no punctuation, no trailing spaces e.g. "auth service" not "Auth Service."
- entity_type: one of person | system | process | milestone | team | concept | document — or a new type if none fit
- relationship: free-text verb phrase describing the directed link e.g. "depends_on", "owned_by", "blocks", "approved_by", "integrates_with"
- weight: float 0.0–1.0 — your confidence this relationship is explicitly stated (not implied)
- Only extract relationships where BOTH entities appear in this text
- If no clear entities or relationships exist, return empty lists

Respond ONLY with valid JSON, no markdown, no preamble:
{
  "entities": [
    {"canonical_name": "...", "entity_type": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "relationship": "...", "weight": 0.95}
  ]
}

Text:
"""


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    """Lowercase + strip punctuation + collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)   # remove punctuation
    name = re.sub(r"\s+", " ", name)       # collapse spaces
    return name.strip()


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_graph_from_chunk(chunk_text: str) -> dict:
    """
    Call Gemini to extract entities + relationships from one parent chunk.

    Returns:
        {
            "entities": [{"canonical_name": str, "entity_type": str}],
            "relationships": [{"source": str, "target": str,
                               "relationship": str, "weight": float}]
        }
    Returns empty lists on any failure — never raises.
    """
    empty = {"entities": [], "relationships": []}

    try:
        prompt   = EXTRACTION_PROMPT + chunk_text
        response = client.models.generate_content(
            model=GEMINI_GEN_MODEL,
            contents=prompt,
        )
        raw = response.text.strip()

        # Strip markdown fences if Gemini adds them despite instructions
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",    "", raw)

        parsed = json.loads(raw)

        # Normalise all entity names before returning
        for e in parsed.get("entities", []):
            e["canonical_name"] = _normalise(e.get("canonical_name", ""))

        for r in parsed.get("relationships", []):
            r["source"] = _normalise(r.get("source", ""))
            r["target"] = _normalise(r.get("target", ""))

        return parsed

    except json.JSONDecodeError as e:
        print(f"[GraphBuilder] JSON parse error: {e} — raw: {raw[:200]}")
        return empty
    except Exception as e:
        print(f"[GraphBuilder] Extraction failed: {e}")
        return empty


# ── DB upsert ─────────────────────────────────────────────────────────────────

def _upsert_entity(cursor, canonical_name: str, entity_type: str,
                   doc_id: int, parent_id: str, now: str) -> int | None:
    """
    Insert entity if it doesn't already exist for this doc.
    Returns the entity id (existing or newly inserted).
    Entities are scoped per doc — same name in two docs = two entity rows,
    allowing edges to correctly reference which document they came from.
    """
    try:
        row = cursor.execute(
            "SELECT id FROM entities WHERE canonical_name = ? AND doc_id = ?",
            (canonical_name, doc_id),
        ).fetchone()

        if row:
            return row[0]

        cursor.execute(
            """INSERT INTO entities (canonical_name, entity_type, doc_id, parent_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (canonical_name, entity_type, doc_id, parent_id, now),
        )
        return cursor.lastrowid

    except Exception as e:
        print(f"[GraphBuilder] Entity upsert failed for '{canonical_name}': {e}")
        return None


def _upsert_edge(cursor, source_id: int, target_id: int,
                 relationship: str, weight: float, doc_id: int, now: str):
    """
    Insert edge if this exact (source, target, relationship, doc_id) combo
    doesn't already exist — prevents duplicates on retro re-runs.
    """
    try:
        existing = cursor.execute(
            """SELECT id FROM edges
               WHERE source_id = ? AND target_id = ? AND relationship = ? AND doc_id = ?""",
            (source_id, target_id, relationship, doc_id),
        ).fetchone()

        if existing:
            return

        cursor.execute(
            """INSERT INTO edges (source_id, target_id, relationship, weight, doc_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source_id, target_id, relationship, weight, doc_id, now),
        )
    except Exception as e:
        print(f"[GraphBuilder] Edge upsert failed ({source_id}→{target_id}): {e}")


# ── Main entry point ──────────────────────────────────────────────────────────

def build_graph_for_document(
    doc_id:      int,
    parent_ids:  list[str],
    parent_texts: list[str],
    doc_title:   str = None,
    conn=None,
):
    """
    Extract and store the knowledge graph for one document.

    Called from indexer.py (new uploads) with the open conn + parent data
    already in memory — no extra DB reads needed.

    Called from retro_graph.py with conn=None — opens its own connection.

    Args:
        doc_id:       document id from documents table
        parent_ids:   list of parent_id strings e.g. ["7_p000", "7_p001"]
        parent_texts: list of parent chunk texts (same order as parent_ids)
        doc_title:    for telemetry logging
        conn:         existing sqlite3 connection (indexer path) or None (retro path)

    Returns:
        dict: {"entities_added": int, "edges_added": int, "chunks_processed": int}
    """
    phase_start  = time.perf_counter()
    own_conn     = conn is None
    entities_added = 0
    edges_added    = 0
    chunks_processed = 0
    now = datetime.now(timezone.utc).isoformat()

    log_event(doc_id, "Graph Extraction Start", doc_title=doc_title, value=0, unit="ms")

    try:
        if own_conn:
            conn = get_db_conn()

        cursor = conn.cursor()

        for parent_id, chunk_text in zip(parent_ids, parent_texts):
            if not chunk_text.strip():
                continue

            # ── Extract from this chunk ───────────────────────────────────────
            extracted = extract_graph_from_chunk(chunk_text)
            entities_in_chunk  = extracted.get("entities",      [])
            relationships      = extracted.get("relationships",  [])

            if not entities_in_chunk:
                chunks_processed += 1
                continue

            # ── Build a name → entity_id map for this chunk ───────────────────
            # Needed to resolve source/target names to DB ids when inserting edges
            name_to_id: dict[str, int] = {}

            for ent in entities_in_chunk:
                cname = ent.get("canonical_name", "").strip()
                etype = ent.get("entity_type",    "concept").strip()

                if not cname:
                    continue

                eid = _upsert_entity(cursor, cname, etype, doc_id, parent_id, now)
                if eid:
                    name_to_id[cname] = eid
                    entities_added += 1

            # ── Insert edges ──────────────────────────────────────────────────
            for rel in relationships:
                source_name = rel.get("source", "").strip()
                target_name = rel.get("target", "").strip()
                relationship = rel.get("relationship", "related_to").strip()
                weight       = float(rel.get("weight", 1.0))

                # Clamp weight to valid range
                weight = max(0.0, min(1.0, weight))

                source_id = name_to_id.get(source_name)
                target_id = name_to_id.get(target_name)

                if not source_id or not target_id:
                    # One of the entities wasn't extracted cleanly — skip edge
                    print(f"[GraphBuilder] Skipping edge '{source_name}' → '{target_name}': entity not found in this chunk")
                    continue

                _upsert_edge(cursor, source_id, target_id, relationship, weight, doc_id, now)
                edges_added += 1

            chunks_processed += 1
            print(f"[GraphBuilder] {parent_id}: {len(entities_in_chunk)} entities, {len(relationships)} relationships")

        # ── Commit if we own the connection (retro path) ──────────────────────
        # Indexer path: indexer.py commits in Phase 7 — we don't double-commit
        if own_conn:
            conn.commit()
            conn.close()

    except Exception as e:
        print(f"[GraphBuilder] build_graph_for_document failed for doc_id={doc_id}: {e}")
        if own_conn and conn:
            try:
                conn.close()
            except Exception:
                pass

    finally:
        phase_ms = (time.perf_counter() - phase_start) * 1000
        log_event(
            doc_id, "Graph Extraction Finish",
            doc_title=doc_title,
            value=phase_ms, unit="ms",
            meta={
                "chunks_processed": chunks_processed,
                "entities_added":   entities_added,
                "edges_added":      edges_added,
            },
        )
        print(
            f"[GraphBuilder] doc_id={doc_id}  chunks={chunks_processed}  "
            f"entities={entities_added}  edges={edges_added}  {phase_ms:.0f}ms"
        )

    return {
        "entities_added":   entities_added,
        "edges_added":      edges_added,
        "chunks_processed": chunks_processed,
    }
