import sqlite3

DB_PATH = "documents.db"

def get_db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_by TEXT NOT NULL,
            content_summary TEXT,
            doc_type TEXT,
            file_path TEXT,
            created_at TEXT
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_title ON documents(title)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_by ON documents(created_by)")
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_search 
        USING fts5(title, content_summary, content);
    ''')
    conn.commit()
    conn.close()
