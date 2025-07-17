import ast
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from sklearn.preprocessing import normalize

model = SentenceTransformer('all-MiniLM-L6-v2')

function_text = []
function_metadata = []

def extract_functions_from_file(file_path: str) -> List[Tuple[str, str, dict]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start_line = node.lineno - 1
            end_line = max(child.lineno for child in ast.walk(node) if hasattr(child, 'lineno'))
            lines = source.splitlines()[start_line:end_line]
            code_block = "\n".join(lines)
            docstring = ast.get_docstring(node) or ""
            full_text = docstring + "\n" + code_block

            metadata = {
                "name": node.name,
                "code": code_block,
                "docstring": docstring,
                "file_path": file_path,
                "start_line": start_line + 1,
                "end_line": end_line
            }

            functions.append((node.name, full_text, metadata))
    
    return functions

def extract_functions_from_folder(folder_path: str) -> List[Tuple[str, str, dict]]:
    all_functions = []
    skip_dirs = {'venv', 'env', 'autofixer-env', '__pycache__', '.git'}
    for root, dirs, files in os.walk(folder_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.abspath(os.path.join(root, file))
                try:
                    functions = extract_functions_from_file(full_path)
                    all_functions.extend(functions)
                except Exception as e:
                    print(f"Failed to extract from {full_path}: {e}")
    return all_functions

def embed_and_store(functions: List[Tuple[str, str, dict]]):
    global function_text, function_metadata

    for _, full_text, metadata in functions:
        function_text.append(full_text)
        function_metadata.append(metadata)

    vectors = model.encode(function_text, convert_to_tensor=False)
    vectors = np.array(vectors).astype("float32")
    vectors = normalize(vectors, axis=1)

    index = faiss.IndexFlatIP(vectors.shape[1])  # Cosine similarity
    index.add(vectors)

    faiss.write_index(index, 'data/function_index.index')
    with open('data/function_metadata.pkl', 'wb') as f:
        pickle.dump(function_metadata, f)

    print(f" Indexed {len(function_text)} functions from the folder.")

def search_functions(query: str, top_k: int = 3):
    index = faiss.read_index('data/function_index.index')
    with open('data/function_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    query_vector = model.encode([query], convert_to_tensor=False)
    query_vector = np.array(query_vector).astype("float32")
    query_vector = normalize(query_vector, axis=1)

    D, I = index.search(query_vector, top_k)

    print(f"\n Results for query: '{query}'\n")
    for i, idx in enumerate(I[0]):
        match = metadata[idx]
        print(f"{i+1}. Function Name: {match['name']}")
        print(f" File: {match['file']}")
        print(f" Lines: {match['start_line']}–{match['end_line']}")
        print(f" Docstring: {match['doc'] or 'No docstring'}")
        print("\n" + match['code'])
        print("—" * 50)

if __name__ == "__main__":
    folder_path = r"C:\Users\Asus\Desktop\autofixer\codebase"  # Use raw string for Windows paths!
    if not os.path.exists(folder_path):
        print(f" Folder '{folder_path}' not found.")
    else:
        funcs = extract_functions_from_folder(folder_path)
        embed_and_store(funcs)
        # search_functions("save user preferences")
