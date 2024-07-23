import fitz  
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from collections import defaultdict

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(texts, model):
    chunks = [chunk_text(text) for text in texts]
    flat_chunks = [chunk for sublist in chunks for chunk in sublist]
    embeddings = model.encode(flat_chunks, convert_to_tensor=True)
    return flat_chunks, embeddings

pdf_paths = [
    "C:/Users/dell/Documents/fiction/01 - The Lost Hero.pdf",
    "C:/Users/dell/Documents/fiction/Tic-Tac-Toe Game.pdf",
    "C:/Users/dell/Documents/fiction/02 - The Son of Neptune.pdf",
    "C:/Users/dell/Documents/fiction/02-The_Dark_Prophecy.pdf" 
]

text_chunks_by_file = defaultdict(list)

texts = []
for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    texts.append(text)
    chunks = chunk_text(text)
    text_chunks_by_file[pdf_path].extend(chunks)

model = SentenceTransformer('all-MiniLM-L6-v2')

flat_chunks, embeddings = generate_embeddings(texts, model)

np.save('texts.npy', np.array(flat_chunks))
np.save('embeddings.npy', embeddings.cpu().numpy())

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.cpu().numpy())

faiss.write_index(index, 'faiss_index.index')

print("Text chunks by file:")
for file_path, chunks in text_chunks_by_file.items():
    print(f"{file_path}: {len(chunks)} chunks")
