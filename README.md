# SimpleText Enterprise

**RAG-powered Document Intelligence Platform (Agent-Ready)**

**Sagar Samrat Das | Principal Product Manager**
GitHub: https://github.com/segadSagar2605/simpletext-enterprise
Demo: https://youtu.be/nD3Az-HeVvw

---

## Overview

SimpleText Enterprise is a document intelligence system that enables users to query large collections of enterprise documents using natural language. It combines hybrid retrieval (keyword + semantic), reranking, and grounded response generation to return precise, context-aware answers.

The system is designed to solve a common enterprise problem: information exists, but cannot be discovered efficiently.

---

## Problem

In most organisations, critical knowledge is stored in PDFs, Word files, and spreadsheets across shared repositories. However:

* Keyword search returns too many irrelevant results
* Users receive documents instead of answers
* Semantic intent is not captured
* Discovery does not scale with growing data

The core hypothesis behind this system:

> Combining semantic retrieval with reranking and grounding LLM responses in retrieved context can make any document corpus reliably queryable.

---

## Approach

The system uses a Retrieval Augmented Generation (RAG) architecture with a hybrid retrieval strategy.

### Retrieval Pipeline

1. Semantic retrieval using vector embeddings
2. Keyword retrieval using BM25 (FTS5)
3. Identity-based fusion of results
4. Reranking using a cross-encoder (FlashRank)
5. Top-ranked context passed to the LLM

This design ensures both:

* Exact matches are not missed
* Semantic queries are correctly interpreted

---

## Architecture

| Component              | Role                                           |
| ---------------------- | ---------------------------------------------- |
| FastAPI Backend        | Orchestrates ingestion, retrieval, and ranking |
| SQLite (B-Tree + FTS5) | Metadata and keyword search                    |
| ChromaDB               | Vector storage and semantic retrieval          |
| Gemini Embeddings      | Converts text into semantic vectors            |
| FlashRank              | Improves precision through reranking           |
| MCP Server             | Enables integration with AI agents             |
| Evaluation Framework   | Measures retrieval and response quality        |

---

## Key Decisions

* **RAG over Fine-Tuning**
  Avoids retraining costs and ensures responses are always grounded in current data

* **Chunking Strategy**
  Recursive chunking with overlap preserves semantic meaning across boundaries

* **Reranking Layer**
  Improves precision by scoring query-document relevance

* **Agent Integration (MCP)**
  Positions the system as a reusable capability in an agent ecosystem

---

## API Overview

| Endpoint     | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `/ask`       | Executes full retrieval pipeline and returns relevant context |
| `/index`     | Processes and indexes uploaded documents                      |
| `/documents` | Lists indexed documents                                       |
| `/preview`   | Returns document summaries                                    |
| `/eval`      | Runs evaluation using LLM-as-judge                            |
| `/metrics`   | Provides performance insights                                 |
| `/health`    | System health check                                           |

---

## Evaluation

The system was tested across a mixed-domain corpus including enterprise documents, technical manuals, and reports.

### Results

* All valid queries (simple, multi-hop, cross-document): **5/5 accuracy**
* Negative queries (information not present): **correctly rejected**

Key observation:
The system distinguishes between related information and correct answers, reducing hallucination risk.

---

## Cost Profile

For a corpus of 5,000 documents and ~50 queries per day:

* Annual cost (local deployment): ~$78
* Per-query cost: <$0.005

The architecture scales efficiently because:

* Indexing is a one-time cost
* Query costs remain low
* No retraining is required

---

## Roadmap

* Persistent conversational memory
* Multi-agent query routing
* Integration of structured metadata with natural language search

---

## Takeaways

This project demonstrates:

* Application of product thinking to AI system design
* Tradeoff management between precision, cost, and latency
* End-to-end ownership from problem definition to evaluation
* Readiness for integration into agent-based ecosystems

---

## Contact

Sagar Samrat Das
Email: [sagar.das@iiml.org](mailto:sagar.das@iiml.org)
LinkedIn: https://linkedin.com/in/sagar-samrat-das
