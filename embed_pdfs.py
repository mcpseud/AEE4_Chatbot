import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdfs_from_directory(directory):
    pdf_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            pdf_texts[filename] = extract_text_from_pdf(filepath)
    return pdf_texts

def extract_text_from_pdf(filepath):
    text = ""
    with fitz.open(filepath) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind(" ", 0, chunk_size)
        if split_at == -1:  # No space found, force split
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:]
    chunks.append(text)
    return chunks

def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

def store_embeddings_with_faiss(embeddings, output_dir='faiss_index'):
    os.makedirs(output_dir, exist_ok=True)
    
    d = embeddings.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # Build the index (L2 distance)

    # Add embeddings to the index
    index.add(embeddings)
    
    # Save the FAISS index
    faiss.write_index(index, os.path.join(output_dir, 'faiss.index'))
    print(f"FAISS index with {index.ntotal} embeddings saved.")

def process_pdf_directory(directory, chunk_size=500):
    pdf_texts = load_pdfs_from_directory(directory)
    
    all_embeddings = []
    for filename, text in pdf_texts.items():
        print(f"Processing {filename}...")
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=chunk_size)
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks)
        all_embeddings.append(embeddings)

    # Combine all embeddings into one array
    all_embeddings = np.vstack(all_embeddings)

    # Store the embeddings in a FAISS index
    store_embeddings_with_faiss(all_embeddings)

    print("Processing complete.")

# usage:
if __name__ == "__main__":
    process_pdf_directory('path_to_pdf_directory')
