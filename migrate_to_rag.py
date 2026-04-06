import os
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from app.services.extraction import extract_text_from_file

# 1. Initialize AI Model and Vector DB
# Loaded model takes ~400MB of your 8GB RAM
model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def migrate():
    # Ensure path points to the root documents.db
    conn = sqlite3.connect('./documents.db')
    cursor = conn.cursor()
    
    # CORRECTED: Using 'title' and 'file_path' - earlier code had 'name' which doesn't exist in the schema
    cursor.execute("SELECT id, title, file_path FROM documents")
    files = cursor.fetchall()

    for file_id, file_name, file_path in files:
        print(f"\n--- Processing: {file_name} ---")
        
        # Pull text using your updated extraction logic
        text = extract_text_from_file(file_path)
        if not text:
            print(f"    Skipping: No text extracted from {file_path}")
            continue

        # 2. Create chunks (text-only, low RAM impact)
        chunks = [text[i:i+500] for i in range(0, len(text), 450)]
        print(f"   Total chunks to index: {len(chunks)}")

        # 3. RAM-SAFE STREAMING BATCHES
        batch_size = 500 
        for i in range(0, len(chunks), batch_size):
            batch_text = chunks[i : i + batch_size]
            
            # Encode ONLY this small batch to keep memory sawtooth low
            batch_embeddings = model.encode(batch_text).tolist()
            
            batch_ids = [f"{file_id}_{j}" for j in range(i, i + len(batch_text))]
            batch_metadatas = [{"file_name": file_name, "db_id": file_id} for _ in range(len(batch_text))]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_text,
                metadatas=batch_metadatas
            )
            print(f"   Indexed chunks {i} to {i + len(batch_text)}")
            
            # Helping Python's Garbage Collector clear RAM
            del batch_embeddings
            del batch_text

    print("\n Migration Success! Your existing data is now vectorized.")
    conn.close()

if __name__ == "__main__":
    migrate()