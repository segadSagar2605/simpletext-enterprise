from app.database import get_db_conn
from app.services.extraction import extract_text_from_file

def background_content_indexing(title: str, content_summary: str, file_path: str):
    full_text = extract_text_from_file(file_path)
    conn = get_db_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO doc_search (title, content_summary, content) VALUES (?, ?, ?)", 
        (title, content_summary, full_text)
    )
    conn.commit()
    conn.close()
