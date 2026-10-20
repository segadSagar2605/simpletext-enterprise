import os
import threading
import uuid
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from flashrank import Ranker, RerankRequest
import chromadb

from app.database import get_db_conn

# Shared clients for retrieval and reranking
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_GEN_MODEL = "gemini-2.5-flash-lite"

ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="/tmp")
ranker_lock = threading.Lock()

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = _chroma_client.get_or_create_collection(name="enterprise_docs")


def rewrite_query_if_needed(query: str, condensed_history: str) -> str:
    if not condensed_history.strip():
        return query

    prompt = f"""You are an query rewriting assistant for an enterprise document search system.

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
            contents=prompt,
        )
        rewritten = response.text.strip()
        return rewritten if rewritten else query
    except Exception as e:
        print(f"[Query Rewriting] Gemini error: {e}")
        return query


def run_ask(
    q: str,
    session_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())
    if conversation_history is None:
        conversation_history = []

    MAX_HISTORY_TURNS = 10
    history = conversation_history[-MAX_HISTORY_TURNS:]
    conn = get_db_conn()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT summary FROM conversation_memory
            WHERE session_id = ? AND role = 'assistant' AND summary IS NOT NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, MAX_HISTORY_TURNS),
        )
        summary_rows = cursor.fetchall()
        condensed_history = " | ".join([row[0] for row in reversed(summary_rows)])
        q = rewrite_query_if_needed(q, condensed_history)

        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=q,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        query_vector = [result.embeddings[0].values]
        vector_results = collection.query(query_embeddings=query_vector, n_results=5)

        clean_q = q.replace('"', ' ').strip()
        if not clean_q:
            keyword_ids = []
        else:
            fts_query = f'"{clean_q}"'
            try:
                cursor.execute(
                    "SELECT parent_id FROM doc_search WHERE doc_search MATCH ? LIMIT 5",
                    (fts_query,),
                )
                keyword_ids = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                print(f"FTS5 Search Error: {e}")
                keyword_ids = []

        candidate_parent_ids = set(keyword_ids)
        for metadata in vector_results['metadatas'][0]:
            candidate_parent_ids.add(metadata['parent_id'])

        passages_for_reranking = []
        for p_id in candidate_parent_ids:
            cursor.execute("SELECT content FROM parents WHERE id = ?", (p_id,))
            row = cursor.fetchone()
            if row:
                passages_for_reranking.append({"id": p_id, "text": row[0]})

        if passages_for_reranking:
            rerank_request = RerankRequest(query=q, passages=passages_for_reranking)
            with ranker_lock:
                results = ranker.rerank(rerank_request)
            top_passages = [r['text'] for r in results[:3]]
            top_scores = [float(round(r['score'], 4)) for r in results[:3]]
        else:
            top_passages = []
            top_scores = []

        now = datetime.now().isoformat()
        assistant_content = "\n\n---\n\n".join(top_passages)

        try:
            summary_prompt = f"""Summarise this Q&A turn in under 100 words.
                            Focus on key entities, decisions, and conclusions only.
                            Return ONLY the summary. No preamble.

                            Q: {q}
                            A: {assistant_content[:2000]}"""
            summary_response = client.models.generate_content(
                model=GEMINI_GEN_MODEL,
                contents=summary_prompt,
            )
            turn_summary = summary_response.text.strip()
        except Exception as e:
            print(f"[Summary Generation] Gemini error: {e}")
            turn_summary = None

        cursor.execute(
            "INSERT INTO conversation_memory (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", q, now),
        )
        cursor.execute(
            "INSERT INTO conversation_memory (session_id, role, content, summary, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", assistant_content, turn_summary, now),
        )
        conn.commit()

        return {
            "query": q,
            "session_id": session_id,
            "retrieved_context": assistant_content,
            "rerank_scores": top_scores,
            "sources_found": len(passages_for_reranking),
            "history": [
                {"role": getattr(h, "role", h.get("role")), "content": getattr(h, "content", h.get("content"))}
                for h in history
            ],
        }
    finally:
        conn.close()
