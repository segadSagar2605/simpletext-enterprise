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
    
    # Enable Write-Ahead Logging for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    return conn

def init_db():
    """
    Initializes the database schema with a Parent-Child hierarchy.
    This is Step 1 of the RAG (Retrieval-Augmented Generation) upgrade.
    """
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # ---  METADATA TABLE (B-Tree Indexing) ---
    # Keeps your dashboard fast for sorting/filtering by title, date, or author.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_by TEXT,
            content_summary TEXT,
            doc_type TEXT,
            file_path TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    
    # Add status column if it doesn't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'Pending';")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # B-Tree indexes for fast dashboard performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON documents(title)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_by ON documents(created_by)")

    # ---  THE PARENT TABLE (Context Store) ---
    # NEW: This stores full, clean paragraphs or sections (Parents).
    # This is the "Source of Truth" Claude will read.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parents (
            id TEXT PRIMARY KEY,       -- Format: doc_id + p_index (e.g., 1_p001)
            doc_id INTEGER,
            content TEXT,              -- Full, logical paragraph text
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    ''')

    # --- VIRTUAL TABLE (FTS5 Keyword Search) ---
    # UPDATED: Links directly to parents for pinpoint accuracy.
    # parent_id is UNINDEXED: we retrieve it but don't waste memory searching it.
    # Removed the 'DROP' command to ensure data persistence
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_search 
        USING fts5(parent_id UNINDEXED, content);
    ''')    
    
    conn.commit()
    conn.close()
    print("Database Blueprint updated successfully with Parent-Child support.")

if __name__ == "__main__":
    init_db()