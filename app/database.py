import os
import sqlite3

# Absolute path to the app/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Absolute path to the project root (one level up from app/)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
# DB always lands in the project root folder regardless of working directory
DB_PATH = os.path.join(PROJECT_ROOT, "documents.db")

def get_db_conn():
    """Ensures the database is always created in the project root with optimized settings."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    """
    Initializes the database schema with a Parent-Child hierarchy.
    Phase 3 adds: entities and edges tables for GraphRAG.
    """
    conn = get_db_conn()
    cursor = conn.cursor()

    # ── DOCUMENTS (B-Tree metadata) ───────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            created_by      TEXT,
            content_summary TEXT,
            doc_type        TEXT,
            file_path       TEXT,
            created_at      TEXT,
            status          TEXT DEFAULT 'Pending'
        )
    ''')
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'Pending';")
    except sqlite3.OperationalError:
        pass

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title      ON documents(title)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_by ON documents(created_by)")

    # ── PARENTS (context store) ───────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parents (
            id     TEXT    PRIMARY KEY,   -- e.g. 1_p001
            doc_id INTEGER,
            content TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    ''')

    # ── FTS5 (keyword search) ─────────────────────────────────────────────────
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_search
        USING fts5(parent_id UNINDEXED, content);
    ''')

    # ── CONVERSATION MEMORY ───────────────────────────────────────────────────
    # summary column: compact turn summary used by query rewriter next turn.
    # Stored at save time so the rewriter never reads raw chunks.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT    NOT NULL,
            user_id    TEXT    NOT NULL DEFAULT 'default',
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            summary    TEXT,
            created_at TEXT    NOT NULL
        )
    ''')
    # Migration — adds summary to existing databases without data loss
    try:
        cursor.execute("ALTER TABLE conversation_memory ADD COLUMN summary TEXT;")
    except sqlite3.OperationalError:
        pass

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON conversation_memory(session_id)")

    # ── ENTITIES (Phase 3 — GraphRAG) ─────────────────────────────────────────
    # One row per unique entity extracted from a document.
    # entity_type: person | system | process | milestone | team | concept
    # doc_id: which document this entity was extracted from
    # parent_id: which parent chunk — links back to full context in parents table
    # canonical_name: normalised form e.g. "auth service" not "Auth Service."
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS entities (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT    NOT NULL,
            entity_type    TEXT    NOT NULL,
            doc_id         INTEGER NOT NULL,
            parent_id      TEXT    NOT NULL,
            created_at     TEXT    NOT NULL,
            FOREIGN KEY (doc_id)    REFERENCES documents (id),
            FOREIGN KEY (parent_id) REFERENCES parents   (id)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_name    ON entities(canonical_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_type    ON entities(entity_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_doc     ON entities(doc_id)")

    # ── EDGES (Phase 3 — GraphRAG) ────────────────────────────────────────────
    # One row per directed relationship between two entities.
    # relationship: free-text label returned by Gemini e.g. depends_on, approved_by,
    #               blocks, replaces, integrates_with — no fixed enum, stored exactly
    #               as extracted and normalised at query time not at index time.
    # weight: confidence score from Gemini extraction (0.0 – 1.0)
    # doc_id: document where this relationship was found
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS edges (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id       INTEGER NOT NULL,
            target_id       INTEGER NOT NULL,
            relationship    TEXT    NOT NULL,
            weight          REAL    NOT NULL DEFAULT 1.0,
            doc_id          INTEGER NOT NULL,
            created_at      TEXT    NOT NULL,
            FOREIGN KEY (source_id) REFERENCES entities (id),
            FOREIGN KEY (target_id) REFERENCES entities (id),
            FOREIGN KEY (doc_id)    REFERENCES documents (id)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_rel    ON edges(relationship)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_doc    ON edges(doc_id)")

    conn.commit()
    conn.close()
    print("Database schema initialised — Phase 3 GraphRAG tables ready.")


if __name__ == "__main__":
    init_db()
