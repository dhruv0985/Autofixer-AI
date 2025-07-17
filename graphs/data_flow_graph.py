import networkx as nx
import ast

def extract_data_flow_edges(code: str, func_name: str) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node(func_name, type='function')

    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for arg in node.args.args:
                G.add_node(arg.arg, type='variable')
                G.add_edge(arg.arg, func_name, weight=1.0, confidence=1.0, dependency_type='arg')

            for n in ast.walk(node):
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if isinstance(target, ast.Name):
                            G.add_node(target.id, type='variable')
                            G.add_edge(target.id, func_name, weight=1.0, confidence=1.0, dependency_type='assign')

                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        G.add_node(n.func.id, type='function')
                        G.add_edge(n.func.id, func_name, weight=1.0, confidence=1.0, dependency_type='call')
                    elif isinstance(n.func, ast.Attribute):
                        G.add_node(n.func.attr, type='function')
                        G.add_edge(n.func.attr, func_name, weight=1.0, confidence=1.0, dependency_type='call')

    return G
