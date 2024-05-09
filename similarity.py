import numpy as np

def search_similar_documents(query_embedding, index, document_store, k=1):
    # Search the index for the k most similar embeddings
    D, I = index.search(np.array([query_embedding], dtype='float32'), k)
    results = [(document_store[i], d) for i, d in zip(I[0], D[0])]
    return results