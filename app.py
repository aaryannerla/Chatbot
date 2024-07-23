from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

app = Flask(__name__)
CORS(app) 

texts = np.load('texts.npy', allow_pickle=True)
embeddings = np.load('embeddings.npy')

index = faiss.read_index('faiss_index.index')

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    question_embedding = model.encode([question], convert_to_tensor=True).cpu().numpy()

    D, I = index.search(question_embedding, k=5) 

    results = [texts[i] for i in I[0]]

    return jsonify({"answer": results[0]}) 

if __name__ == '__main__':
    app.run(debug=True)
