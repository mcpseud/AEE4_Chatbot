import os
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import openai

# Initialize the FastAPI app
app = FastAPI()

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
index = faiss.read_index('pdf_embeddings.index')

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the request body for the FastAPI endpoint
class QueryRequest(BaseModel):
    week: int
    query: str

# Define the function to search the FAISS index
def search_faiss(query_embedding, top_k=5):
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, top_k)
    return indices[0]

# Define the function to call GPT-4o
def generate_response(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {query}"},
        ]
    )
    return response['choices'][0]['message']['content']

# Define the FastAPI endpoint
@app.post("/api/chat")
async def chat(query_request: QueryRequest):
    # Embed the user's query
    query_embedding = model.encode(query_request.query)

    # Retrieve the most relevant chunks from the FAISS index
    relevant_indices = search_faiss(query_embedding)
    
    # Retrieve the actual text chunks
    chunks = []
    for idx in relevant_indices:
        chunk_text = f"Text chunk {idx}"  # Placeholder, replace with actual text retrieval
        chunks.append(chunk_text)
    
    # Combine the relevant chunks into a context string
    context = " ".join(chunks)

    # Generate a response using GPT-4
    response_text = generate_response(query_request.query, context)

    return {"response": response_text}

# Command to run the server: uvicorn main:app --reload
