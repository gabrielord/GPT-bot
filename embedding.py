from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def text_to_embedding(text, model_embedding):
    return model_embedding.encode([text])[0]

def create_faiss_index(dimension):
    return faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity

def add_document_to_store(filename, text, embedding, index, document_store):
    # Add the document to the FAISS index
    idx = len(document_store)  # Get new index position based on current store size
    index.add(np.array([embedding], dtype='float32'))  # Add embedding to FAISS index
    # Store metadata with the same index
    document_store[idx] = {
        'filename': filename,
        'text': text,
        'embedding': embedding
    }