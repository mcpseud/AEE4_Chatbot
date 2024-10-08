from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
import os

app = Flask(__name__)

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index
try:
    index = faiss.read_index("embeddings/faiss.index")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    index = None

# Load text chunks associated with the embeddings
try:
    with open('embeddings/text_chunks.txt', 'r') as f:
        text_chunks = f.read().split("\n----\n")
except Exception as e:
    print(f"Error loading text chunks: {e}")
    text_chunks = []

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_query = data.get('message', '').strip()
        history = data.get('history', [])

        if not user_query:
            return jsonify({"error": "Empty query received"}), 400

        # Step 1: Embed the user's query
        query_embedding = model.encode([user_query])

        # Step 2: Search for relevant text chunks using FAISS
        if index is None or not text_chunks:
            return jsonify({"error": "Internal error: FAISS index or text chunks not properly initialized"}), 500
        
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = index.search(query_embedding, k)
        
        # Check if any relevant text chunks were found
        if len(indices[0]) == 0:
            return jsonify({"error": "No relevant text found"}), 404
        
        # Gather the most relevant text chunks
        relevant_texts = [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]
        
        if not relevant_texts:
            return jsonify({"error": "No relevant text found"}), 404
        
        # Combine the relevant texts into a single context
        context = "\n".join(relevant_texts)
        
        # Step 3: Prepare the full conversation for GPT-4o, including history
        messages = history + [
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": user_query},
        ]

        # Call GPT-4o with the conversation history and context
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages
        )
        
        # Extract the response from GPT-4o
        gpt_response = response['choices'][0]['message']['content']
        
        # Step 4: Return the response to the frontend
        return jsonify({"response": gpt_response})
    
    except openai.error.OpenAIError as e:
        # Handle errors from the OpenAI API (or specific to GPT-4o)
        return jsonify({"error": f"Error communicating with GPT-4o: {str(e)}"}), 500
    
    except Exception as e:
        # Handle any other unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
