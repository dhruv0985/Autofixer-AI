import networkx as nx
from graphs.call_analysis import get_called_functions

def build_call_graph(functions: list) -> nx.DiGraph:
    """
    Builds a full call graph for the whole codebase.
    Takes a list of {'name':..., 'code':...} dicts.
    """
    G = nx.DiGraph()

    for func in functions:
        caller = func['name']
        G.add_node(caller, type='function')

        called = get_called_functions(func['code'])
        for callee in called:
            G.add_node(callee, type='function')
            G.add_edge(caller, callee, weight=1.0, confidence=1.0, call_count=1, is_recursive=False)

    return G
