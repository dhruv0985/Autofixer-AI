from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

model = SentenceTransformer('all-MiniLM-L6-v2')

def build_semantic_graph(functions: list, threshold=0.7) -> nx.DiGraph:
    """
    Build a semantic similarity graph from function names.
    """
    G = nx.DiGraph()
    names = [f['name'] for f in functions]
    embeddings = model.encode(names)

    sim_matrix = cosine_similarity(embeddings)

    for i in range(len(names)):
        G.add_node(names[i], type='function')
        for j in range(i + 1, len(names)):
            score = sim_matrix[i][j]
            if score >= threshold:
                G.add_node(names[j], type='function')
                G.add_edge(names[i], names[j], weight=1.0, confidence=1.0, relationship='similar', similarity=score)

    return G
