import sqlite3

def get_db_conn():
    conn = sqlite3.connect("documents.db")
    return conn

def init_db():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # Standard table for Metadata (Uses B-Tree indexing)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_by TEXT,
            content_summary TEXT,
            doc_type TEXT,
            file_path TEXT,
            created_at TEXT
        )
    ''')

    # Explicitly creating B-Tree indexes for dashboard performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON documents(title)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_by ON documents(created_by)")

    # Virtual table for Deep Content Search (Uses FTS5 tokenization)
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_search 
        USING fts5(title, content_summary, content);
    ''')
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()