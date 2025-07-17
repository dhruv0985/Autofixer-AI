import networkx as nx
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict
# from .call_graph import build_call_graph
# from .data_flow_graph import build_data_flow_graph
# from .semantic_graph import build_semantic_graph

class NodeType(Enum):
    FUNCTION = "function"
    VARIABLE = "variable"
    CLASS = "class"
    MODULE = "module"
    CONDITION = "condition"
    LOOP = "loop"
    EXCEPTION = "exception"

class EdgeType(Enum):
    CALLS = "calls"
    DATA_FLOW = "data_flow"
    SEMANTIC = "semantic"
    CONTROLS = "controls"
    DEFINES = "defines"
    USES = "uses"
    INHERITS = "inherits"
    DEPENDS_ON = "depends_on"

@dataclass
class NodeAttributes:
    """Unified node attributes combining all graph types"""
    node_id: str
    node_type: NodeType
    name: str
    file_path: str
    line_number: int
    
    # Call graph attributes
    call_frequency: int = 0
    is_entry_point: bool = False
    call_depth: int = 0
    
    # Data flow attributes
    data_dependencies: List[str] = None
    variable_scope: str = None
    data_type: str = None
    
    # Semantic attributes
    semantic_role: str = None
    complexity_score: float = 0.0
    bug_probability: float = 0.0
    related_concepts: List[str] = None
    
    def __post_init__(self):
        if self.data_dependencies is None:
            self.data_dependencies = []
        if self.related_concepts is None:
            self.related_concepts = []

@dataclass
class EdgeAttributes:
    """Unified edge attributes combining all graph types"""
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    
    # Call graph edge attributes
    call_count: int = 0
    is_recursive: bool = False
    
    # Data flow edge attributes
    data_dependency_type: str = None  # "read", "write", "modify"
    variable_name: str = None
    
    # Semantic edge attributes
    semantic_relationship: str = None
    similarity_score: float = 0.0

class GraphMerger:
    """
    Merges call graph, data flow graph, and semantic graph into a unified representation
    """
    
    def __init__(self):
        self.unified_graph = nx.MultiDiGraph()
        self.node_mapping = {}  # Maps original node IDs to unified node IDs
        self.edge_relationships = defaultdict(list)
        
    def merge_graphs(self, call_graph: nx.DiGraph, data_flow_graph: nx.DiGraph, 
                    semantic_graph: nx.DiGraph) -> nx.MultiDiGraph:
        """
        Main method to merge all three graphs
        """
        print("Starting graph merger...")
        
        # Step 1: Create unified node set
        self._create_unified_nodes(call_graph, data_flow_graph, semantic_graph)
        
        # Step 2: Merge edges with proper typing
        self._merge_call_edges(call_graph)
        self._merge_data_flow_edges(data_flow_graph)
        self._merge_semantic_edges(semantic_graph)
        
        # Step 3: Add cross-graph relationships
        self._add_cross_graph_relationships()
        
        # Step 4: Calculate composite metrics
        self._calculate_composite_metrics()
        
        # Step 5: Validate merged graph
        self._validate_merged_graph()
        
        print(f"Graph merger completed. Nodes: {len(self.unified_graph.nodes)}, Edges: {len(self.unified_graph.edges)}")
        return self.unified_graph
    
    def _create_unified_nodes(self, call_graph: nx.DiGraph, data_flow_graph: nx.DiGraph, 
                             semantic_graph: nx.DiGraph):
        """Create unified node set by merging nodes from all graphs"""
        all_nodes = {}
        
        # Process call graph nodes
        for node_id, attrs in call_graph.nodes(data=True):
            unified_attrs = NodeAttributes(
                node_id=node_id,
                node_type=NodeType(attrs.get('type', 'function')),
                name=attrs.get('name', ''),
                file_path=attrs.get('file_path', ''),
                line_number=attrs.get('line_number', 0),
                call_frequency=attrs.get('call_frequency', 0),
                is_entry_point=attrs.get('is_entry_point', False),
                call_depth=attrs.get('call_depth', 0)
            )
            all_nodes[node_id] = unified_attrs
        
        # Process data flow graph nodes
        for node_id, attrs in data_flow_graph.nodes(data=True):
            if node_id in all_nodes:
                # Merge attributes
                all_nodes[node_id].data_dependencies = attrs.get('dependencies', [])
                all_nodes[node_id].variable_scope = attrs.get('scope', None)
                all_nodes[node_id].data_type = attrs.get('data_type', None)
            else:
                # Create new node
                unified_attrs = NodeAttributes(
                    node_id=node_id,
                    node_type=NodeType(attrs.get('type', 'variable')),
                    name=attrs.get('name', ''),
                    file_path=attrs.get('file_path', ''),
                    line_number=attrs.get('line_number', 0),
                    data_dependencies=attrs.get('dependencies', []),
                    variable_scope=attrs.get('scope', None),
                    data_type=attrs.get('data_type', None)
                )
                all_nodes[node_id] = unified_attrs
        
        # Process semantic graph nodes
        for node_id, attrs in semantic_graph.nodes(data=True):
            if node_id in all_nodes:
                # Merge attributes
                all_nodes[node_id].semantic_role = attrs.get('role', None)
                all_nodes[node_id].complexity_score = attrs.get('complexity', 0.0)
                all_nodes[node_id].bug_probability = attrs.get('bug_probability', 0.0)
                all_nodes[node_id].related_concepts = attrs.get('related_concepts', [])
            else:
                # Create new node
                unified_attrs = NodeAttributes(
                    node_id=node_id,
                    node_type=NodeType(attrs.get('type', 'function')),
                    name=attrs.get('name', ''),
                    file_path=attrs.get('file_path', ''),
                    line_number=attrs.get('line_number', 0),
                    semantic_role=attrs.get('role', None),
                    complexity_score=attrs.get('complexity', 0.0),
                    bug_probability=attrs.get('bug_probability', 0.0),
                    related_concepts=attrs.get('related_concepts', [])
                )
                all_nodes[node_id] = unified_attrs
        
        # Add nodes to unified graph
        for node_id, attrs in all_nodes.items():
            self.unified_graph.add_node(node_id, **attrs.__dict__)
            self.node_mapping[node_id] = node_id
    
    def _merge_call_edges(self, call_graph: nx.DiGraph):
        """Merge call graph edges"""
        for source, target, attrs in call_graph.edges(data=True):
            edge_attrs = EdgeAttributes(
                edge_type=EdgeType.CALLS,
                weight=attrs.get('weight', 1.0),
                confidence=attrs.get('confidence', 1.0),
                call_count=attrs.get('call_count', 1),
                is_recursive=attrs.get('is_recursive', False)
            )
            
            self.unified_graph.add_edge(source, target, **edge_attrs.__dict__)
    
    def _merge_data_flow_edges(self, data_flow_graph: nx.DiGraph):
        """Merge data flow graph edges"""
        for source, target, attrs in data_flow_graph.edges(data=True):
            edge_attrs = EdgeAttributes(
                edge_type=EdgeType.DATA_FLOW,
                weight=attrs.get('weight', 1.0),
                confidence=attrs.get('confidence', 1.0),
                data_dependency_type=attrs.get('dependency_type', 'read'),
                variable_name=attrs.get('variable_name', None)
            )
            
            self.unified_graph.add_edge(source, target, **edge_attrs.__dict__)
    
    def _merge_semantic_edges(self, semantic_graph: nx.DiGraph):
        """Merge semantic graph edges"""
        for source, target, attrs in semantic_graph.edges(data=True):
            edge_attrs = EdgeAttributes(
                edge_type=EdgeType.SEMANTIC,
                weight=attrs.get('weight', 1.0),
                confidence=attrs.get('confidence', 1.0),
                semantic_relationship=attrs.get('relationship', None),
                similarity_score=attrs.get('similarity', 0.0)
            )
            
            self.unified_graph.add_edge(source, target, **edge_attrs.__dict__)
    
    def _add_cross_graph_relationships(self):
        """Add relationships that exist across different graph types"""
        # Find functions that both call each other AND share data
        call_edges = [(u, v) for u, v, d in self.unified_graph.edges(data=True) 
                     if d['edge_type'] == EdgeType.CALLS]
        data_edges = [(u, v) for u, v, d in self.unified_graph.edges(data=True) 
                     if d['edge_type'] == EdgeType.DATA_FLOW]
        
        # Add dependency relationships for functions with both call and data relationships
        for call_edge in call_edges:
            for data_edge in data_edges:
                if call_edge[0] == data_edge[0] and call_edge[1] == data_edge[1]:
                    # Strong dependency - both call and data relationship
                    dep_attrs = EdgeAttributes(
                        edge_type=EdgeType.DEPENDS_ON,
                        weight=2.0,
                        confidence=0.9
                    )
                    self.unified_graph.add_edge(call_edge[0], call_edge[1], **dep_attrs.__dict__)
    
    def _calculate_composite_metrics(self):
        """Calculate composite metrics using advanced graph algorithms"""
        # 1. PageRank for importance scoring
        pagerank_scores = self._calculate_pagerank_importance()
        
        # 2. Topological sort for fix ordering
        fix_order = self._calculate_topological_fix_order()
        
        # 3. Strongly Connected Components for circular dependencies
        scc_analysis = self._analyze_strongly_connected_components()
        
        # 4. Centrality measures for critical nodes
        centrality_scores = self._calculate_centrality_measures()
        
        # 5. Community detection for related function clusters
        communities = self._detect_function_communities()
        
        # Combine all metrics
        for node_id in self.unified_graph.nodes():
            node_attrs = self.unified_graph.nodes[node_id]
            
            # Advanced importance score using PageRank + Centrality
            pagerank_importance = pagerank_scores.get(node_id, 0) * 0.4
            centrality_importance = centrality_scores.get(node_id, 0) * 0.3
            structural_importance = node_attrs.get('call_frequency', 0) * 0.3
            
            importance_score = pagerank_importance + centrality_importance + structural_importance
            self.unified_graph.nodes[node_id]['importance_score'] = importance_score
            self.unified_graph.nodes[node_id]['pagerank_score'] = pagerank_scores.get(node_id, 0)
            self.unified_graph.nodes[node_id]['centrality_score'] = centrality_scores.get(node_id, 0)
            
            # Enhanced bug risk score
            structural_risk = min(node_attrs.get('call_depth', 0) / 10.0, 1.0) * 0.2
            complexity_risk = node_attrs.get('complexity_score', 0) * 0.3
            semantic_risk = node_attrs.get('bug_probability', 0) * 0.2
            scc_risk = scc_analysis.get(node_id, {}).get('risk_factor', 0) * 0.3
            
            bug_risk_score = structural_risk + complexity_risk + semantic_risk + scc_risk
            self.unified_graph.nodes[node_id]['bug_risk_score'] = bug_risk_score
            
            # Add algorithmic metadata
            self.unified_graph.nodes[node_id]['fix_order'] = fix_order.get(node_id, 999)
            self.unified_graph.nodes[node_id]['scc_id'] = scc_analysis.get(node_id, {}).get('component_id', -1)
            self.unified_graph.nodes[node_id]['community_id'] = communities.get(node_id, -1)
    
    def _calculate_pagerank_importance(self) -> Dict[str, float]:
        """Use PageRank to find most important nodes across all graph types"""
        # Create weighted graph for PageRank
        weighted_graph = nx.DiGraph()
        
        for node in self.unified_graph.nodes():
            weighted_graph.add_node(node)
        
        # Add edges with composite weights
        for source, target, attrs in self.unified_graph.edges(data=True):
            edge_type = attrs.get('edge_type')
            base_weight = attrs.get('weight', 1.0)
            confidence = attrs.get('confidence', 1.0)
            
            # Weight edges differently based on type
            if edge_type == EdgeType.CALLS:
                weight = base_weight * 1.5 * confidence  # Call relationships are important
            elif edge_type == EdgeType.DATA_FLOW:
                weight = base_weight * 1.2 * confidence  # Data flow is moderately important
            elif edge_type == EdgeType.SEMANTIC:
                weight = base_weight * 0.8 * confidence  # Semantic similarity is supporting
            elif edge_type == EdgeType.DEPENDS_ON:
                weight = base_weight * 2.0 * confidence  # Dependencies are critical
            else:
                weight = base_weight * confidence
            
            if weighted_graph.has_edge(source, target):
                weighted_graph[source][target]['weight'] += weight
            else:
                weighted_graph.add_edge(source, target, weight=weight)
        
        # Calculate PageRank with custom weights
        try:
            pagerank = nx.pagerank(weighted_graph, weight='weight', alpha=0.85, max_iter=1000)
            return pagerank
        except:
            # Fallback to unweighted PageRank
            return nx.pagerank(weighted_graph, alpha=0.85, max_iter=1000)
    
    def _calculate_topological_fix_order(self) -> Dict[str, int]:
        """Calculate optimal fix order using topological sort"""
        # Create dependency graph for topological sort
        dependency_graph = nx.DiGraph()
        
        for node in self.unified_graph.nodes():
            dependency_graph.add_node(node)
        
        # Add edges that represent "must fix before" relationships
        for source, target, attrs in self.unified_graph.edges(data=True):
            edge_type = attrs.get('edge_type')
            
            if edge_type == EdgeType.CALLS:
                # Called functions should be fixed before callers
                dependency_graph.add_edge(target, source)
            elif edge_type == EdgeType.DATA_FLOW:
                # Data producers should be fixed before consumers
                dependency_graph.add_edge(source, target)
            elif edge_type == EdgeType.DEPENDS_ON:
                # Dependencies should be fixed first
                dependency_graph.add_edge(source, target)
        
        # Handle cycles using SCC
        fix_order = {}
        try:
            # Get topological order
            topo_order = list(nx.topological_sort(dependency_graph))
            for i, node in enumerate(topo_order):
                fix_order[node] = i
        except nx.NetworkXError:
            # Graph has cycles, use SCC-based approach
            sccs = list(nx.strongly_connected_components(dependency_graph))
            scc_graph = nx.condensation(dependency_graph, sccs)
            
            order = 0
            for scc_id in nx.topological_sort(scc_graph):
                scc_nodes = sccs[scc_id]
                # Within SCC, order by importance
                sorted_nodes = sorted(scc_nodes, 
                                    key=lambda n: self.unified_graph.nodes[n].get('importance_score', 0), 
                                    reverse=True)
                for node in sorted_nodes:
                    fix_order[node] = order
                    order += 1
        
        return fix_order
    
    def _analyze_strongly_connected_components(self) -> Dict[str, Dict[str, Any]]:
        """Analyze SCCs to identify circular dependencies and fix challenges"""
        scc_analysis = {}
        
        # Create call+dependency graph for SCC analysis
        scc_graph = nx.DiGraph()
        for node in self.unified_graph.nodes():
            scc_graph.add_node(node)
        
        for source, target, attrs in self.unified_graph.edges(data=True):
            edge_type = attrs.get('edge_type')
            if edge_type in [EdgeType.CALLS, EdgeType.DEPENDS_ON, EdgeType.DATA_FLOW]:
                scc_graph.add_edge(source, target)
        
        # Find strongly connected components
        sccs = list(nx.strongly_connected_components(scc_graph))
        
        for i, scc in enumerate(sccs):
            scc_size = len(scc)
            
            # Calculate risk factor based on SCC size and complexity
            if scc_size == 1:
                risk_factor = 0.1  # No circular dependency
            elif scc_size <= 3:
                risk_factor = 0.3  # Small cycle - manageable
            elif scc_size <= 6:
                risk_factor = 0.6  # Medium cycle - challenging
            else:
                risk_factor = 0.9  # Large cycle - high risk
            
            # Analyze internal complexity
            internal_edges = 0
            for node1 in scc:
                for node2 in scc:
                    if scc_graph.has_edge(node1, node2):
                        internal_edges += 1
            
            complexity_factor = min(internal_edges / (scc_size * scc_size), 1.0)
            final_risk = risk_factor * (1 + complexity_factor)
            
            for node in scc:
                scc_analysis[node] = {
                    'component_id': i,
                    'component_size': scc_size,
                    'risk_factor': final_risk,
                    'internal_edges': internal_edges,
                    'fix_strategy': self._get_scc_fix_strategy(scc_size)
                }
        
        return scc_analysis
    
    def _get_scc_fix_strategy(self, scc_size: int) -> str:
        """Determine fix strategy based on SCC size"""
        if scc_size == 1:
            return "direct_fix"
        elif scc_size <= 3:
            return "sequential_fix"
        elif scc_size <= 6:
            return "batch_fix"
        else:
            return "refactor_required"
    
    def _calculate_centrality_measures(self) -> Dict[str, float]:
        """Calculate various centrality measures to identify critical nodes"""
        centrality_scores = {}
        
        # Create undirected graph for centrality calculations
        undirected_graph = self.unified_graph.to_undirected()
        
        try:
            # Betweenness centrality - nodes that are bridges
            betweenness = nx.betweenness_centrality(undirected_graph)
            
            # Closeness centrality - nodes close to all others
            closeness = nx.closeness_centrality(undirected_graph)
            
            # Degree centrality - nodes with many connections
            degree = nx.degree_centrality(undirected_graph)
            
            # Eigenvector centrality - nodes connected to important nodes
            try:
                eigenvector = nx.eigenvector_centrality(undirected_graph, max_iter=1000)
            except:
                eigenvector = {node: 0 for node in undirected_graph.nodes()}
            
            # Combine centrality measures
            for node in undirected_graph.nodes():
                combined_centrality = (
                    betweenness.get(node, 0) * 0.3 +
                    closeness.get(node, 0) * 0.2 +
                    degree.get(node, 0) * 0.2 +
                    eigenvector.get(node, 0) * 0.3
                )
                centrality_scores[node] = combined_centrality
                
        except Exception as e:
            print(f"Warning: Centrality calculation failed: {e}")
            # Fallback to simple degree centrality
            centrality_scores = nx.degree_centrality(undirected_graph)
        
        return centrality_scores
    
    def _detect_function_communities(self) -> Dict[str, int]:
        """Detect communities of related functions using modularity"""
        communities = {}
        
        try:
            # Create undirected graph for community detection
            undirected_graph = self.unified_graph.to_undirected()
            
            # Use Louvain algorithm for community detection
            # Note: This requires python-louvain package
            # For now, we'll use a simple approach based on connected components
            
            # Get weakly connected components as communities
            connected_components = list(nx.connected_components(undirected_graph))
            
            for i, component in enumerate(connected_components):
                for node in component:
                    communities[node] = i
                    
        except Exception as e:
            print(f"Warning: Community detection failed: {e}")
            # Fallback: each node is its own community
            communities = {node: i for i, node in enumerate(self.unified_graph.nodes())}
        
        return communities
    
    def _validate_merged_graph(self):
        """Validate the merged graph for consistency"""
        # Check for orphaned nodes
        isolated_nodes = list(nx.isolates(self.unified_graph))
        if isolated_nodes:
            print(f"Warning: Found {len(isolated_nodes)} isolated nodes")
        
        # Check for self-loops
        self_loops = list(nx.selfloop_edges(self.unified_graph))
        if self_loops:
            print(f"Found {len(self_loops)} self-loops")
        
        # Validate edge types
        edge_types = set()
        for _, _, attrs in self.unified_graph.edges(data=True):
            edge_types.add(attrs.get('edge_type'))
        
        print(f"Edge types in merged graph: {edge_types}")
    
    def get_node_context(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive context for a node from all graph types"""
        if node_id not in self.unified_graph.nodes:
            return {}
        
        node_attrs = self.unified_graph.nodes[node_id]
        
        # Get all related nodes
        predecessors = list(self.unified_graph.predecessors(node_id))
        successors = list(self.unified_graph.successors(node_id))
        
        # Group by edge type
        context = {
            'node_attributes': node_attrs,
            'callers': [],
            'callees': [],
            'data_sources': [],
            'data_targets': [],
            'semantic_relations': [],
            'dependencies': []
        }
        
        for pred in predecessors:
            edges = self.unified_graph.get_edge_data(pred, node_id)
            for edge_data in edges.values():
                if edge_data['edge_type'] == EdgeType.CALLS:
                    context['callers'].append(pred)
                elif edge_data['edge_type'] == EdgeType.DATA_FLOW:
                    context['data_sources'].append(pred)
                elif edge_data['edge_type'] == EdgeType.SEMANTIC:
                    context['semantic_relations'].append(pred)
                elif edge_data['edge_type'] == EdgeType.DEPENDS_ON:
                    context['dependencies'].append(pred)
        
        for succ in successors:
            edges = self.unified_graph.get_edge_data(node_id, succ)
            for edge_data in edges.values():
                if edge_data['edge_type'] == EdgeType.CALLS:
                    context['callees'].append(succ)
                elif edge_data['edge_type'] == EdgeType.DATA_FLOW:
                    context['data_targets'].append(succ)
        
        return context
    
    def find_critical_paths(self, start_node: str, end_node: str = None) -> List[List[str]]:
        """Find critical paths using advanced graph algorithms"""
        if end_node is None:
            # Find all paths from start_node to high-risk nodes
            high_risk_nodes = [n for n, attrs in self.unified_graph.nodes(data=True) 
                              if attrs.get('bug_risk_score', 0) > 0.7]
            paths = []
            for risk_node in high_risk_nodes:
                try:
                    node_paths = list(nx.all_simple_paths(self.unified_graph, start_node, risk_node, cutoff=10))
                    paths.extend(node_paths)
                except nx.NetworkXNoPath:
                    continue
            return paths
        else:
            try:
                return list(nx.all_simple_paths(self.unified_graph, start_node, end_node, cutoff=10))
            except nx.NetworkXNoPath:
                return []
    
    def get_optimal_fix_sequence(self, bug_nodes: List[str]) -> List[Dict[str, Any]]:
        """Generate optimal fix sequence using topological sort and SCC analysis"""
        fix_sequence = []
        
        # Get fix order for all bug nodes
        bug_fix_order = []
        for node in bug_nodes:
            node_attrs = self.unified_graph.nodes[node]
            bug_fix_order.append({
                'node': node,
                'fix_order': node_attrs.get('fix_order', 999),
                'scc_id': node_attrs.get('scc_id', -1),
                'importance': node_attrs.get('importance_score', 0),
                'risk': node_attrs.get('bug_risk_score', 0),
                'fix_strategy': node_attrs.get('fix_strategy', 'direct_fix')
            })
        
        # Sort by fix order, then by importance
        bug_fix_order.sort(key=lambda x: (x['fix_order'], -x['importance']))
        
        # Group by SCC for batch processing
        scc_groups = defaultdict(list)
        for item in bug_fix_order:
            scc_groups[item['scc_id']].append(item)
        
        # Process each SCC group
        for scc_id, scc_nodes in scc_groups.items():
            if len(scc_nodes) == 1:
                # Single node - direct fix
                fix_sequence.append({
                    'type': 'single_fix',
                    'nodes': [scc_nodes[0]['node']],
                    'strategy': 'direct_fix',
                    'priority': scc_nodes[0]['importance']
                })
            else:
                # Multiple nodes in SCC - need careful ordering
                strategy = scc_nodes[0]['fix_strategy']
                if strategy == 'refactor_required':
                    fix_sequence.append({
                        'type': 'refactor',
                        'nodes': [n['node'] for n in scc_nodes],
                        'strategy': 'refactor_required',
                        'priority': max(n['importance'] for n in scc_nodes)
                    })
                else:
                    fix_sequence.append({
                        'type': 'batch_fix',
                        'nodes': [n['node'] for n in scc_nodes],
                        'strategy': strategy,
                        'priority': max(n['importance'] for n in scc_nodes)
                    })
        
        return fix_sequence
    
    def find_impact_radius(self, node: str, radius: int = 3) -> Dict[str, List[str]]:
        """Find all nodes within impact radius using BFS"""
        impact_radius = {
            'direct_impact': [],      # Distance 1
            'secondary_impact': [],   # Distance 2
            'tertiary_impact': [],    # Distance 3+
            'all_impacted': []
        }
        
        try:
            # BFS to find nodes at different distances
            distances = nx.single_source_shortest_path_length(self.unified_graph, node, cutoff=radius)
            
            for target_node, distance in distances.items():
                if target_node == node:
                    continue
                    
                impact_radius['all_impacted'].append(target_node)
                
                if distance == 1:
                    impact_radius['direct_impact'].append(target_node)
                elif distance == 2:
                    impact_radius['secondary_impact'].append(target_node)
                else:
                    impact_radius['tertiary_impact'].append(target_node)
                    
        except Exception as e:
            print(f"Warning: Impact radius calculation failed: {e}")
        
        return impact_radius
    
    def get_fix_dependencies(self, node: str) -> Dict[str, List[str]]:
        """Get all dependencies that must be considered when fixing a node"""
        dependencies = {
            'must_fix_first': [],     # Dependencies that must be fixed first
            'must_fix_together': [],  # Nodes in same SCC
            'must_validate': [],      # Nodes that must be validated after fix
            'may_impact': []          # Nodes that may be impacted
        }
        
        node_attrs = self.unified_graph.nodes[node]
        scc_id = node_attrs.get('scc_id', -1)
        
        # Find nodes in same SCC
        if scc_id >= 0:
            for other_node, attrs in self.unified_graph.nodes(data=True):
                if attrs.get('scc_id') == scc_id and other_node != node:
                    dependencies['must_fix_together'].append(other_node)
        
        # Find dependencies based on topological order
        node_fix_order = node_attrs.get('fix_order', 999)
        
        for other_node, attrs in self.unified_graph.nodes(data=True):
            other_fix_order = attrs.get('fix_order', 999)
            
            # Check if there's a direct dependency
            if self.unified_graph.has_edge(other_node, node):
                edge_data = self.unified_graph.get_edge_data(other_node, node)
                for edge_attrs in edge_data.values():
                    if edge_attrs.get('edge_type') == EdgeType.DEPENDS_ON:
                        dependencies['must_fix_first'].append(other_node)
                        break
            
            # Check if we need to validate this node after fix
            if self.unified_graph.has_edge(node, other_node):
                edge_data = self.unified_graph.get_edge_data(node, other_node)
                for edge_attrs in edge_data.values():
                    if edge_attrs.get('edge_type') in [EdgeType.CALLS, EdgeType.DATA_FLOW]:
                        dependencies['must_validate'].append(other_node)
                        break
        
        # Find potential impact using PageRank and centrality
        impact_radius = self.find_impact_radius(node, radius=2)
        dependencies['may_impact'] = impact_radius['direct_impact'] + impact_radius['secondary_impact']
        
        return dependencies
    
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """Comprehensive graph analysis using multiple algorithms"""
        analysis = {
            'basic_stats': {
                'num_nodes': len(self.unified_graph.nodes),
                'num_edges': len(self.unified_graph.edges),
                'density': nx.density(self.unified_graph),
                'is_connected': nx.is_weakly_connected(self.unified_graph)
            },
            'algorithmic_analysis': {},
            'critical_nodes': {},
            'fix_recommendations': {}
        }
        
        # PageRank analysis
        pagerank_scores = nx.pagerank(self.unified_graph)
        top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis['algorithmic_analysis']['top_pagerank_nodes'] = top_pagerank
        
        # SCC analysis
        sccs = list(nx.strongly_connected_components(self.unified_graph))
        analysis['algorithmic_analysis']['num_sccs'] = len(sccs)
        analysis['algorithmic_analysis']['largest_scc_size'] = max(len(scc) for scc in sccs) if sccs else 0
        analysis['algorithmic_analysis']['circular_dependencies'] = [list(scc) for scc in sccs if len(scc) > 1]
        
        # Centrality analysis
        try:
            betweenness = nx.betweenness_centrality(self.unified_graph)
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis['algorithmic_analysis']['top_betweenness_nodes'] = top_betweenness
        except:
            analysis['algorithmic_analysis']['top_betweenness_nodes'] = []
        
        # Critical nodes identification
        for node, attrs in self.unified_graph.nodes(data=True):
            if attrs.get('bug_risk_score', 0) > 0.7 and attrs.get('importance_score', 0) > 0.5:
                analysis['critical_nodes'][node] = {
                    'bug_risk': attrs.get('bug_risk_score', 0),
                    'importance': attrs.get('importance_score', 0),
                    'pagerank': pagerank_scores.get(node, 0),
                    'scc_id': attrs.get('scc_id', -1)
                }
        
        # Fix recommendations
        analysis['fix_recommendations'] = self._generate_fix_recommendations()
        
        return analysis
    
    def _generate_fix_recommendations(self) -> Dict[str, Any]:
        """Generate fix recommendations based on graph analysis"""
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'refactor_candidates': []
        }
        
        for node, attrs in self.unified_graph.nodes(data=True):
            bug_risk = attrs.get('bug_risk_score', 0)
            importance = attrs.get('importance_score', 0)
            scc_id = attrs.get('scc_id', -1)
            
            # Determine priority
            if bug_risk > 0.8 and importance > 0.6:
                recommendations['high_priority'].append({
                    'node': node,
                    'reason': 'High bug risk and high importance',
                    'risk': bug_risk,
                    'importance': importance
                })
            elif bug_risk > 0.6 or importance > 0.7:
                recommendations['medium_priority'].append({
                    'node': node,
                    'reason': 'Moderate bug risk or high importance',
                    'risk': bug_risk,
                    'importance': importance
                })
            elif bug_risk > 0.4:
                recommendations['low_priority'].append({
                    'node': node,
                    'reason': 'Low to moderate bug risk',
                    'risk': bug_risk,
                    'importance': importance
                })
            
            # Check if node is in large SCC (refactor candidate)
            if scc_id >= 0:
                scc_size = sum(1 for _, a in self.unified_graph.nodes(data=True) if a.get('scc_id') == scc_id)
                if scc_size > 5:
                    recommendations['refactor_candidates'].append({
                        'node': node,
                        'scc_size': scc_size,
                        'reason': 'Part of large circular dependency'
                    })
        
        return recommendations
        """Export the merged graph for visualization or further analysis"""
        # Convert to JSON-serializable format
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        for node_id, attrs in self.unified_graph.nodes(data=True):
            node_data = {'id': node_id, 'attributes': attrs}
            graph_data['nodes'].append(node_data)
        
        for source, target, attrs in self.unified_graph.edges(data=True):
            edge_data = {'source': source, 'target': target, 'attributes': attrs}
            graph_data['edges'].append(edge_data)
        
        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2, default=str)
        
        print(f"Merged graph exported to {output_path}")

# Example usage with advanced algorithms
# if __name__ == "__main__":
#     # Initialize merger
#     merger = GraphMerger()
    
    # Assuming you have your three graphs ready
    # call_graph = load_call_graph()
    # data_flow_graph = load_data_flow_graph()
    # semantic_graph = load_semantic_graph()
    
    # Merge graphs with advanced algorithms
    # unified_graph = merger.merge_graphs(call_graph, data_flow_graph, semantic_graph)
    
    # ALGORITHMIC ANALYSIS:
    
    # 1. PageRank Analysis - Find most important functions
    # analysis = merger.analyze_graph_structure()
    # print("Top PageRank nodes:", analysis['algorithmic_analysis']['top_pagerank_nodes'])
    
    # 2. Topological Sort - Get optimal fix order
    # bug_nodes = ['func1', 'func2', 'func3']
    # fix_sequence = merger.get_optimal_fix_sequence(bug_nodes)
    # print("Fix sequence:", fix_sequence)
    
    # 3. SCC Analysis - Find circular dependencies
    # circular_deps = analysis['algorithmic_analysis']['circular_dependencies']
    # print("Circular dependencies:", circular_deps)
    
    # 4. Centrality Analysis - Find bridge functions
    # bridge_nodes = analysis['algorithmic_analysis']['top_betweenness_nodes']
    # print("Bridge nodes:", bridge_nodes)
    
    # 5. Impact Analysis - Find what gets affected by a fix
    # impact = merger.find_impact_radius('problematic_function', radius=3)
    # print("Impact radius:", impact)
    
    # 6. Dependency Analysis - What must be fixed together
    # deps = merger.get_fix_dependencies('problematic_function')
    # print("Fix dependencies:", deps)
    
    # 7. Critical Path Analysis
    # paths = merger.find_critical_paths('entry_point')
    # print("Critical paths:", paths)
    
    # 8. Get comprehensive context
    # context = merger.get_node_context("problematic_function")
    # print("Node context:", context)
    
    # 9. Get fix recommendations
    # recommendations = analysis['fix_recommendations']
    # print("Fix recommendations:", recommendations)
    
    # Export for visualization
    # merger.export_merged_graph("unified_graph.json")
    
    # print("Advanced graph merger with algorithms ready!")
    # print("Algorithms used:")
    # print("- PageRank: For importance scoring")
    # print("- Topological Sort: For fix ordering")
    # print("- Strongly Connected Components: For circular dependency detection")
    # print("- Centrality Measures: For critical node identification")
    # print("- Community Detection: For related function clustering")
    # print("- BFS: For impact radius calculation")