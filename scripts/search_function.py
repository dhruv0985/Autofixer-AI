import faiss
import pickle
from sklearn.preprocessing import normalize 
from sentence_transformers import SentenceTransformer
import numpy as np


model=SentenceTransformer('all-MiniLM-L6-v2')

index=faiss.read_index("function_index.index")

with open("function_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search_codebase(query: str, top_k: int =3):
    query_vec = model.encode([query], convert_to_tensor=False)
    query_vec = np.array(query_vec).astype("float32")
    query_vec = normalize(query_vec, axis=1)
    D, I=index.search(query_vec.reshape(1, -1), top_k)

    print(f"\nResults for query: '{query}'\n")
    for i, idx in enumerate(I[0]):
        func= metadata[idx]
        print(f"{i+1}. Function: {func['name']}")
        print(f"   File: {func['file']}")
        print(f"   Lines: {func['start_line']}–{func['end_line']}")
        print(f"   Docstring: {func['docstring'] or 'No docstring'}\n")
        print(func['code'])
        print("—" * 50)

if __name__ == "__main__":
    print(" AutoFixer Semantic Function Search")
    while True:
        query = input("\n Enter a bug/issue description (or 'exit' to quit):\n> ")
        if query.lower() in ["exit", "quit"]:
            break
        search_codebase(query)