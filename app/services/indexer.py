import uuid
from .extraction import extract_text_from_file
from ..database import get_db_conn
import chromadb
from sentence_transformers import SentenceTransformer

# Load model once (approx 400MB RAM usage)
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def recursive_splitter(text, max_size=1000):
    """
    The 'Smart Scissors': Splitting text by paragraphs, then sentences.
    Ensures thoughts like headers and rows stay together.
    """
    if len(text) <= max_size:
        return [text]
    
    # Try splitting by paragraph (double newline)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) <= max_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk: 
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    
    if current_chunk: 
        chunks.append(current_chunk.strip())
    return chunks

def background_content_indexing(doc_id: int, file_path: str):
    """
    Step 1: Parent-Child Indexing Engine.
    Populates SQLite Parents, FTS5, and ChromaDB Children.
    """
    full_text = extract_text_from_file(file_path)
    if not full_text:
        print(f" No text found in {file_path}")
        return

    conn = get_db_conn()
    cursor = conn.cursor()

    # 1. Create Parents (~1000 chars)
    # These are the clean blocks Claude will read later.
    parent_blocks = recursive_splitter(full_text, max_size=1000)

    for idx, p_text in enumerate(parent_blocks):
        parent_id = f"{doc_id}_p{idx:03d}"
        
        # Save to SQLite Parent Table (The Source of Truth)
        cursor.execute(
            "INSERT INTO parents (id, doc_id, content) VALUES (?, ?, ?)",
            (parent_id, doc_id, p_text)
        )
        
        # Save to FTS5 (The Keyword Librarian)
        cursor.execute(
            "INSERT INTO doc_search (parent_id, content) VALUES (?, ?)",
            (parent_id, p_text)
        )

        # 2. Create Children (~300 chars) for Vectors
        # These are the 'Sensors' that find the right Parent.
        # We overlap them by 50 chars so we don't miss anything at the edges.
        child_chunks = [p_text[i:i+300] for i in range(0, len(p_text), 250)]
        
        child_embeddings = model.encode(child_chunks).tolist()
        child_ids = [f"{parent_id}_c{j}" for j in range(len(child_chunks))]
        child_metadatas = [{"parent_id": parent_id, "doc_id": doc_id} for _ in child_chunks]

        # Save to ChromaDB (The Mathematician)
        collection.add(
            ids=child_ids,
            embeddings=child_embeddings,
            documents=child_chunks,
            metadatas=child_metadatas
        )

    conn.commit()
    conn.close()
    print(f" Document {doc_id} indexed with {len(parent_blocks)} parents.")