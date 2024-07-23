from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
import os

app = Flask(__name__)
CORS(app)

texts = np.load('texts.npy', allow_pickle=True)
embeddings = np.load('embeddings.npy')

index = faiss.read_index('faiss_index.index')

model = SentenceTransformer('all-MiniLM-L6-v2')

openai.api_key = os.getenv('OPENAI_API_KEY')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    question_embedding = model.encode([question], convert_to_tensor=True).cpu().numpy()

    D, I = index.search(question_embedding, k=5)

    results = [texts[i] for i in I[0]]

    # Use GPT-3.5 to format the answer
    formatted_answer = format_answer_with_gpt3(results[0], question)

    return jsonify({"answer": formatted_answer})

def format_answer_with_gpt3(answer, question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Question: {question}\n\nAnswer: {answer}\n\nPlease format the answer in a clear and concise manner:",
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

if __name__ == '__main__':
    app.run(debug=True)
