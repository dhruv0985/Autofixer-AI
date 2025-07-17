import networkx as nx
from graphs.graph_merger import GraphMerger
from agents.planner_agent import PlannerAgent
from agents.fixer_agent import generate_fix
from agents.validator_agent import validate_fix
from utils.patch_function import patch_function_in_file
from utils.embedding import search_codebase


class CoordinatorAgent:
    """
    Orchestrates the entire multi-agent fix flow with robust planning:
    - merges graphs
    - analyzes structure
    - falls back to embedding search if needed
    - plans optimal fix order with PlannerAgent
    - coordinates fixes, validation, confirmation & patching
    """

    def __init__(self, call_graph, data_flow_graph, semantic_graph, bug_description, metadata=None):
        self.call_graph = call_graph
        self.data_flow_graph = data_flow_graph
        self.semantic_graph = semantic_graph
        self.bug_description = bug_description
        self.graph_merger = GraphMerger()
        self.unified_graph = None
        self.planner = None
        self.function_cache = {}
        self.all_metadata = metadata or []

    def _cache_matches(self, matches):
        """Caches fallback matches for quick lookup."""
        for m in matches:
            if 'file_path' not in m or not m['file_path']:
                m['file_path'] = "unknown.py"
            self.function_cache[m['name']] = m

    def _get_function_code(self, name):
        """Retrieve function metadata with guaranteed file_path."""
        if name in self.function_cache:
            return self.function_cache[name]

        for func_meta in self.all_metadata:
            if func_meta['name'] == name:
                if 'file_path' not in func_meta or not func_meta['file_path']:
                    func_meta['file_path'] = "unknown.py"
                self.function_cache[name] = func_meta
                return func_meta

        return None

    def run(self):
        print("Starting Coordinator Agent with Robust Planning...")

        print("Starting graph merger...")
        self.unified_graph = self.graph_merger.merge_graphs(
            self.call_graph,
            self.data_flow_graph,
            self.semantic_graph
        )

        self.analysis = self.graph_merger.analyze_graph_structure()
        bug_nodes = list(self.analysis['critical_nodes'].keys())

        if not bug_nodes:
            print("No critical bugs detected. Using fallback...")
            matches, all_metadata = search_codebase(self.bug_description)
            if not matches:
                print("No fallback matches found. Exiting.")
                return

            self.all_metadata = all_metadata or []
            self._cache_matches(matches)

            fallback_nodes = list({m['name'] for m in matches})

            for m in matches:
                if m['name'] in self.call_graph:
                    callees = list(self.call_graph.successors(m['name']))
                    for c in callees:
                        if c not in fallback_nodes:
                            fallback_nodes.append(c)
                            callee_meta = self._get_function_code(c)
                            if callee_meta:
                                self.function_cache[c] = callee_meta
                            else:
                                # Add dummy entry to avoid repeated lookups
                                self.function_cache[c] = {
                                    'name': c,
                                    'code': f"# TODO: code for {c} not found",
                                    'file_path': "unknown.py"
                                }
                                print(f"⚠️ No metadata found for callee {c}, using fallback.")

            bug_nodes = fallback_nodes

        print(f"Nodes selected for fixing: {bug_nodes}")

        self.planner = PlannerAgent(
            self.call_graph,
            self.data_flow_graph,
            self.semantic_graph,
            self.unified_graph
        )

        print(f"Creating fix plan for functions: {bug_nodes}")
        fix_batches = self.planner.create_fix_plan(bug_nodes, self.bug_description)
        if not fix_batches:
            print("No fix plan created. Exiting.")
            return

        print(f"Created {len(fix_batches)} fix batches:")

        for batch_idx, batch in enumerate(fix_batches, start=1):
            print(f"\n--- Executing Batch {batch_idx}/{len(fix_batches)} ---")

            for plan in batch:
                print(f"Processing {plan.function_name} using {plan.strategy} strategy")

                main_func = self._get_function_code(plan.function_name)
                if not main_func:
                    print(f"❌ Could not find code for {plan.function_name}")
                    continue

                helpers = [
                    self._get_function_code(h) for h in plan.context_functions
                    if self._get_function_code(h)
                ]

                prompt = self.planner.build_fix_prompt(
                    plan,
                    main_func['code'],
                    helpers,
                    self.bug_description
                )

                fixed_code = generate_fix(
                    instruction=prompt,
                    main_function=main_func['code'],
                    helper_functions=helpers
                )

                validation = validate_fix(fixed_code, main_func.get('file_path', 'unknown.py'))

                if validation['all_pass']:
                    patch_function_in_file(
                        {
                            'name': plan.function_name,
                            'file': main_func.get('file_path', 'unknown.py'),
                            'issue': self.bug_description
                        },
                        fixed_code
                    )
                    print(f"✅ Fix for {plan.function_name} applied.")
                else:
                    print(f"[mypy output]:\n{validation}")
                    print("❌ Validation failed.")
                    if not validation['pytest']:
                        print("\nNo tests found. Here's the generated fix:\n")
                        print("="*40)
                        print(fixed_code)
                        print("="*40)
                        confirm = input("Apply this patch? [Y/N]: ").strip().lower()
                        if confirm == "y":
                            patch_function_in_file(
                                {
                                    'name': plan.function_name,
                                    'file': main_func.get('file_path', 'unknown.py'),
                                    'issue': self.bug_description
                                },
                                fixed_code
                            )
                            print(f"✅ Patch for {plan.function_name} force-applied.")
                        else:
                            print(f"❌ Patch for {plan.function_name} skipped by user.")

        print("\n✅ Coordinator finished.")
