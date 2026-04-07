# SimpleText Enterprise Ingestion Engine

A modular Python framework for document ingestion, metadata management, and full-text search.

## Features
- **Multi-format Support**: PDF, Word (docx), and Excel (xlsx).
- **Hybrid Indexing**: Sync B-Tree for metadata and Async FTS5 for content.
- **Modular Architecture**: Separated services for extraction and indexing.

## Setup & Run
1. **Prepare Lab**: `python -m venv venv`
2. **Activate**: `.\venv\Scripts\activate`
3. **Install**: `pip install -r requirements.txt`
4. **Launch**: `uvicorn app.main:app --reload`

## Project Roadmap
- Modular Backend with a simple and Refined UI
- Vector Embeddings (ChromaDB) for RAG - GEMINI KEY
- MCP (Model Context Protocol) Integration
- Docker Containerization
- Splunk/ELK Audit Traceability
