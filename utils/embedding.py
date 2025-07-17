import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index and metadata
index = faiss.read_index("data/function_index.index")
with open("data/function_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# Semantic search to get the most relevant function
def search_codebase(query: str, top_k: int = 1):
    query_vec = model.encode([query], convert_to_tensor=False)
    query_vec = np.array(query_vec).astype("float32")
    query_vec = normalize(query_vec, axis=1)
    D, I = index.search(query_vec, top_k)

    matches = []
    for i in I[0]:
        m = metadata[i]
        matches.append({
            "name": m["name"],
            "file_path": m["file_path"], 
            "code": m["code"] 
        })

    return matches, metadata