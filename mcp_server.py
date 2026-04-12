"""
SimpleText Enterprise — MCP Server
Exposes the RAG 'ask' capability to Claude Desktop.

Tool exposed:
    ask  —  semantic search + reranking against the Enterprise Knowledge Hub
"""

import httpx
from mcp.server.fastmcp import FastMCP

# ── MCP server instance ───────────────────────────────────────────────────────
mcp = FastMCP(
    name="SimpleText Enterprise Knowledge Hub",
    instructions=(
        "You have access to the SimpleText Enterprise Knowledge Hub — "
        "a RAG system containing ingested enterprise documents. "
        "Use the 'ask' tool to semantically search and retrieve context "
        "from those documents before answering any question about them. "
        "Always use the tool first. Never answer from memory alone."
    ),
)

# ── Backend URL ───────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000"


# ── Tool: list_documents ──────────────────────────────────────────────────────
@mcp.tool()
def list_documents() -> str:
    """
    List all documents in the Enterprise Knowledge Hub with their metadata.

    Use this tool when:
    - The user asks a broad question and you need to discover what documents exist
    - You need to decide WHICH document to search in before calling ask()
    - The user asks "what do you have?", "what files are available?",
      "summarise everything", "what do you know about X project?"
    - You need to check document status, author, type, or date added
    - You want to verify freshness of information before answering

    After calling this, use ask() to retrieve specific content from relevant documents.

    Returns:
        List of all documents with id, title, author, type, status,
        date added, and AI-generated summary.
    """
    try:
        response = httpx.get(
            f"{BACKEND_URL}/get-docs",
            timeout=10.0,
        )
        response.raise_for_status()
        docs = response.json()

        if not docs:
            return "The Knowledge Hub is empty — no documents have been ingested yet."

        lines = [f"[Knowledge Hub — {len(docs)} document(s) available]\n"]
        for doc in docs:
            summary = doc.get('summary', '').strip() or "No summary available"
            lines.append(
                f"• [{doc['id']}] {doc['title']}\n"
                f"  Type: {doc['type']}  |  Author: {doc['author']}  "
                f"|  Status: {doc['status']}  |  Added: {doc['date']}\n"
                f"  Summary: {summary}\n"
            )

        return "\n".join(lines)

    except httpx.ConnectError:
        return (
            "Cannot reach the SimpleText backend at http://127.0.0.1:8000. "
            "Please ensure the FastAPI server is running."
        )
    except Exception as e:
        return f"Knowledge Hub error: {str(e)}"


# ── Tool: ask ─────────────────────────────────────────────────────────────────
@mcp.tool()
def ask(question: str) -> str:
    """
    Semantically search the Enterprise Knowledge Hub and retrieve relevant context.

    Use this tool whenever the user asks anything that may be answered
    by the documents in the Knowledge Hub — regardless of topic, domain,
    or document type.

    Always use this tool before answering. Never rely on memory alone.

    Args:
        question: The natural language question to search for.

    Returns:
        Retrieved context passages from the most relevant documents,
        along with the number of sources found.
    """
    try:
        response = httpx.post(
            f"{BACKEND_URL}/ask",
            json={"q": question},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        retrieved = data.get("retrieved_context", "").strip()
        sources   = data.get("sources_found", 0)

        if not retrieved:
            return "No relevant content found in the Knowledge Hub for that question."

        return (
            f"[Knowledge Hub — {sources} source(s) found]\n\n"
            f"{retrieved}"
        )

    except httpx.ConnectError:
        return (
            "Cannot reach the SimpleText backend at http://127.0.0.1:8000. "
            "Please ensure the FastAPI server is running."
        )
    except httpx.TimeoutException:
        return "The Knowledge Hub took too long to respond. Please try again."
    except Exception as e:
        return f"Knowledge Hub error: {str(e)}"


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
