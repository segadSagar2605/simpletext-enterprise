import importlib


def test_rag_service_and_agent_are_connected():
    rag = importlib.import_module("app.services.rag")
    assert hasattr(rag, "run_ask")

    graph = importlib.import_module("app.agent.graph")
    assert hasattr(graph, "rag_node")
    assert hasattr(graph, "run_agent")
