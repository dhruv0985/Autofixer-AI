import pickle
from graphs.call_graph import build_call_graph
from graphs.data_flow_graph import extract_data_flow_edges
from graphs.semantic_graph import build_semantic_graph
from agents.coordinator_agent import CoordinatorAgent
import networkx as nx
from graphs.call_graph import build_call_graph
from graphs.data_flow_graph import extract_data_flow_edges   
from graphs.semantic_graph import build_semantic_graph


if __name__ == "__main__":
    print("=== AutoFixer Orchestrator ===")



    # 1. Load the function metadata
    with open("data/function_metadata.pkl", "rb") as f:
        functions = pickle.load(f)

    print(f"Loaded {len(functions)} functions from metadata.")

    # 2. Build the Call Graph
    call_graph = build_call_graph(functions)
    print(f"Call Graph: {len(call_graph.nodes)} nodes, {len(call_graph.edges)} edges")

    # 3. Build the Data Flow Graph
    data_flow_graph = nx.DiGraph()
    for func in functions:
        g = extract_data_flow_edges(func['code'], func['name'])
        data_flow_graph.add_nodes_from(g.nodes(data=True))
        data_flow_graph.add_edges_from(g.edges(data=True))
    print(f"Data Flow Graph: {len(data_flow_graph.nodes)} nodes, {len(data_flow_graph.edges)} edges")

    # 4. Build the Semantic Graph
    semantic_graph = build_semantic_graph(functions)
    print(f"Semantic Graph: {len(semantic_graph.nodes)} nodes, {len(semantic_graph.edges)} edges")

    bug_description = input("Describe the bug:\n> ")

    # 5. Run the Coordinator Agent
    coordinator = CoordinatorAgent(call_graph, data_flow_graph, semantic_graph,bug_description)
    coordinator.run()
