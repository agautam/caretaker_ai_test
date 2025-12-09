import chromadb
from chromadb.config import Settings
import re

def split_into_chunks(text, words_per_chunk=100):
    """Split text into chunks of approximately the specified number of words."""
    # Split text into words
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)
    
    return chunks

def main():
    # Read manual.txt
    with open('manual.txt', 'r', encoding='utf-8') as f:
        manual_text = f.read()
    
    # Split text into chunks of 100 words
    chunks = split_into_chunks(manual_text, words_per_chunk=100)
    
    # Create a persistent ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get the collection named 'hvac_manual'
    collection = client.get_or_create_collection(name="hvac_manual")
    
    # Prepare documents and IDs for adding to collection
    documents = chunks
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add chunks to the collection
    collection.add(
        documents=documents,
        ids=ids
    )
    
    print('Manual ingested successfully')

if __name__ == "__main__":
    main()

