import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory containing PDFs
pdf_directory = './pdfs'

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to chunk text into smaller pieces
def chunk_text(text, max_chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + len(word) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to embed chunks and store them
def embed_and_store_chunks(chunks, index):
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    embeddings = embeddings.cpu().detach().numpy()

    # Add embeddings to the index
    index.add(embeddings)

# Initialize FAISS index
embedding_dim = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

# Process each PDF in the directory
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f'Processing {pdf_file}...')
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Embed and store the chunks
        embed_and_store_chunks(chunks, index)

# Save the FAISS index to a file
faiss.write_index(index, 'pdf_embeddings.index')
print('All PDFs processed and embeddings stored.')
