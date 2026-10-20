import os                                    # read API keys from environment
import uuid
from typing import TypedDict                 # structured state definition
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END  # graph container and terminal node
from tavily import TavilyClient              # web search API for external queries
from google import genai                     # Gemini for routing and synthesis
from langgraph.types import Send             # type for sending messages between nodes, used for parallel execution

load_dotenv()

from app.services.rag import run_ask

# ============ AGENT STATE ============
# Shared state object that flows through every node in the graph.
# Each node reads what it needs and writes back its output.
# Fields are added as new nodes are introduced across levels.

class AgentState(TypedDict):
    query:            str   # user's question
    routing_decision: str   # 'rag', 'web_search', or 'both'
    rag_result:       str   # output from RAG node
    rag_error:        str   # clear error marker if document search fails
    web_result:       str   # output from web search node
    final_answer:     str   # synthesised answer
    session_id:       str   # for memory

# ============ CLIENT SETUP ============
# Initialise Gemini and Tavily clients once at module level.
# Reused across all nodes — no re-initialisation per request.

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

GEMINI_GEN_MODEL = "gemini-2.5-flash-lite"  # consistent with main.py


# ============ ROUTER NODE ============
# First node in the graph — classifies user query intent to route flow.

def router_node(state: AgentState) -> dict:
    query = state["query"]  # read query from state

    # Ask Gemini to classify intent — returns one of three routing decisions
    prompt = f"""You are an intent classifier for an enterprise document search system.

Given this user query:
{query}

Classify the intent into exactly one of these three options:
- "rag"        → query needs internal enterprise documents only
- "web_search" → query needs current external information only  
- "both"       → query needs both internal documents and external search

Rules:
- Return ONLY the option string. No explanation. No punctuation.
- "rag" for anything about internal projects, risks, budgets, milestones
- "web_search" for current regulations, market data, recent news
- "both" for comparisons between internal data and external information
- When uncertain between "rag" and "both", always choose "both"

Answer:"""

    response = gemini_client.models.generate_content(
        model=GEMINI_GEN_MODEL,
        contents=prompt
    )
    decision = response.text.strip().lower()

    # Safety — if Gemini returns unexpected value, default to "both"
    if decision not in ["rag", "web_search", "both"]:
        decision = "both"

    return {"routing_decision": decision}


# ============ RAG NODE ============
# Calls the existing /ask endpoint to retrieve internal document context.
# Passes session_id to maintain memory chain across turns.

def rag_node(state: AgentState) -> dict:
    query = state["query"]
    session_id = state.get("session_id") or ""

    try:
        data = run_ask(
            q=query,
            session_id=session_id,
            doc_type=None,
            conversation_history=[]
        )
        return {
            "rag_result": data.get("retrieved_context", ""),
            "rag_error": "",
            "session_id": data.get("session_id", session_id),
        }

    except Exception as e:
        print(f"[RAG Node] Error: {e}")
        return {
            "rag_result": "",
            "rag_error": "Document search failed.",
            "session_id": session_id,
        }

# ============ WEB SEARCH NODE ============
# Calls Tavily to retrieve current external information.
# Stateless — no session or memory needed.

def web_search_node(state: AgentState) -> dict:
    query = state["query"]

    try:
        results = tavily_client.search(query, max_results=5)
        combined = "\n\n".join([r["content"] for r in results.get("results", [])])
        return {"web_result": combined}

    except Exception as e:
        print(f"[Web Search Node] Error: {e}")
        return {"web_result": ""}


# ============ SYNTHESISER NODE ============
# Combines RAG and web results into a clean final answer.
# Only fires when both results are available.

def synthesiser_node(state: AgentState) -> dict:
    query = state["query"]
    rag_result = state.get("rag_result", "")
    web_result = state.get("web_result", "")
    rag_error = state.get("rag_error", "")

    if rag_error:
        if web_result:
            failure_header = "Document search failed. I can answer from the web context below, but the internal document lookup did not succeed.\n\n"
            web_context = web_result[:3000]
            prompt = f"""You are an expert enterprise analyst.

User question: {query}

Important: The internal document search failed. This answer should clearly state that and still answer using the web context only.

{failure_header}

External web context:
{web_context}

Provide a concise answer and clearly mark that the document search failed.
Answer:"""
        else:
            return {"final_answer": "Document search failed. I could not retrieve the internal enterprise documents for this question."}
    else:
        prompt = f"""You are an expert enterprise analyst.

User question: {query}

Internal document context:
{rag_result[:3000]}

External web context:
{web_result[:3000]}

Provide a clear, structured answer using both sources.
Cite which information came from internal documents vs external sources.
Be concise and actionable.

Answer:"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_GEN_MODEL,
            contents=prompt
        )
        answer = response.text.strip()
        if rag_error and web_result:
            return {"final_answer": f"Document search failed.\n\n{answer}"}
        return {"final_answer": answer}

    except Exception as e:
        print(f"[Synthesiser Node] Error: {e}")
        if rag_error and web_result:
            return {"final_answer": "Document search failed.\n\n" + (web_result[:1000] or "Web search results were unavailable.")}
        return {"final_answer": rag_result or web_result}

# ============ GRAPH ASSEMBLY ============
# Wire all nodes and edges together.
# route_parallel enables true parallel execution for "both" routing.
# Synthesiser automatically waits for all incoming nodes to complete.

def route_parallel(state: AgentState):
    decision = state["routing_decision"]
    if decision == "rag":
        return ["rag"]
    elif decision == "web_search":
        return ["web_search"]
    elif decision == "both":
        return ["rag", "web_search"]  # true parallel — both fire simultaneously
    return ["rag"]  # safe default

def build_graph():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesiser", synthesiser_node)

    # Entry point — always start at router
    graph.set_entry_point("router")

    # Conditional edges with parallel support
    # route_parallel returns a list — LangGraph fires all nodes in the list
    graph.add_conditional_edges(
        "router",
        route_parallel,
        ["rag", "web_search"]  # all possible destinations
    )

    # Both RAG and web_search converge at synthesiser
    # Synthesiser waits for all incoming nodes before firing
    graph.add_edge("rag", "synthesiser")
    graph.add_edge("web_search", "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()

# Compile once at module level — reused across all requests
agent_graph = build_graph()


# ============ RUN AGENT ============
# Entry point called from main.py or MCP server.
# Initialises full state and returns clean result dict.

def run_agent(query: str, session_id: str = None) -> dict:
    effective_session_id = session_id or str(uuid.uuid4())
    result = agent_graph.invoke({
        "query": query,
        "session_id": effective_session_id,
        "routing_decision": "",
        "rag_result": "",
        "rag_error": "",
        "web_result": "",
        "final_answer": ""
    })
    return {
        "query": query,
        "routing_decision": result["routing_decision"],
        "final_answer": result["final_answer"],
        "session_id": result.get("session_id", effective_session_id)
    }