import networkx as nx
from graphs.graph_merger import GraphMerger
from agents.planner_agent import PlannerAgent
from agents.fixer_agent import generate_fix
from agents.validator_agent import validate_fix
from utils.patch_function import patch_function_in_file
from utils.embedding import search_codebase
from utils.reindex import reindex_codebase
import subprocess
import tempfile
import os


class CoordinatorAgent:
    """
    Enhanced Orchestrator with iterative self-correction and retry logic.
    - merges graphs
    - analyzes structure
    - falls back to embedding search if needed
    - plans optimal fix order with PlannerAgent
    - coordinates fixes with retry mechanism
    - validates and applies patches
    - handles post-reindex validation
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
        
        # Enhanced retry configuration
        self.max_retries = 3
        self.retry_feedback = {}
        self.reindex_occurred = False
        self.post_reindex_validation = True

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

    def _auto_format_code(self, code: str) -> str:
        """Automatically format code using black."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run black on the file
            result = subprocess.run(
                ["black", temp_file],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Read the formatted code
                with open(temp_file, 'r') as f:
                    formatted_code = f.read()
                
                print("  ✅ Code automatically formatted with black")
                return formatted_code
            else:
                print(f"  ⚠️ Black formatting failed: {result.stderr}")
                return code
                
        except Exception as e:
            print(f"  ⚠️ Auto-formatting error: {e}")
            return code
        finally:
            # Clean up
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.unlink(temp_file)

    def _collect_validation_feedback(self, validation_result, function_name):
        """Collect feedback from validation failures for retry."""
        feedback = []
        
        if not validation_result['syntax_check']:
            feedback.append("Fix syntax errors in the code")
        
        if not validation_result['mypy']:
            feedback.append("Fix type checking issues reported by mypy")
        
        if not validation_result['black']:
            feedback.append("Fix code formatting issues")
        
        if validation_result['pytest_status'] == 'no_tests':
            feedback.append("No tests available for validation")
        elif not validation_result['pytest']:
            feedback.append("Fix issues causing test failures")
        
        return feedback

    def _build_retry_prompt(self, plan, original_code, helpers, attempt_number, feedback):
        """Build an enhanced prompt with retry feedback."""
        retry_context = f"""
RETRY ATTEMPT {attempt_number}/{self.max_retries}

Previous attempt failed with these issues:
{chr(10).join(f"- {issue}" for issue in feedback)}

Please address these specific issues while fixing the original bug.
"""
        
        base_prompt = self.planner.build_fix_prompt(
            plan, original_code, helpers, self.bug_description
        )
        
        # Insert retry context before the rules
        if "RULES:" in base_prompt:
            parts = base_prompt.split("RULES:")
            enhanced_prompt = parts[0] + retry_context + "\nRULES:" + parts[1]
        else:
            enhanced_prompt = base_prompt + "\n\n" + retry_context
        
        return enhanced_prompt

    def _validate_and_apply_fix(self, plan, fixed_code, main_func, attempt, is_final=False):
        """Validate and potentially apply a fix with enhanced logic."""
        # Auto-format the code first
        formatted_code = self._auto_format_code(fixed_code)
        
        # Validate the formatted code
        validation = validate_fix(formatted_code, main_func.get('file_path', 'unknown.py'))
        final_code = validation.get('formatted_code') if validation.get('formatted_code') else formatted_code
        if validation['all_pass']:
            # Perfect - apply the fix
            patch_function_in_file(
                {
                    'name': plan.function_name,
                    'file': main_func.get('file_path', 'unknown.py'),
                    'issue': self.bug_description
                },
                final_code
            )
            print(f"   Fix for {plan.function_name} applied successfully")
            
            # Reindex and mark that it occurred
            print("   Reindexing codebase...")
            reindex_codebase('codebase')
            self.reindex_occurred = True
            print("   Codebase reindexed successfully")
            
            return True, []
        
        # Validation failed - handle based on context
        feedback = self._collect_validation_feedback(validation, plan.function_name)
        
        # Special handling for final attempt
        if is_final and validation['pytest_status'] == 'no_tests':
            # Only formatting issues and no tests - ask user
            if validation['syntax_check'] and validation['mypy']:
                print(f"\n[Final attempt] Validation summary for {plan.function_name}:")
                print(f"  - Syntax: ✅")
                print(f"  - MyPy: ✅")
                print(f"  - Black: {'✅' if validation['black'] else '❌'}")
                print(f"  - Tests: No tests available")
                
                print("\nFormatted fix:")
                print("="*50)
                print(formatted_code)
                print("="*50)
                
                confirm = input("Apply this patch? [Y/N]: ").strip().lower()
                if confirm == "y":
                    patch_function_in_file(
                        {
                            'name': plan.function_name,
                            'file': main_func.get('file_path', 'unknown.py'),
                            'issue': self.bug_description
                        },
                        formatted_code
                    )
                    print(f"  ✅ Patch for {plan.function_name} applied")
                    
                    # Reindex and mark that it occurred
                    print("   Reindexing codebase...")
                    reindex_codebase('codebase')
                    self.reindex_occurred = True
                    print("  ✅ Codebase reindexed successfully")
                    
                    return True, []
                else:
                    print(f"  ❌ Patch for {plan.function_name} skipped by user")
                    return False, feedback
        
        print(f"  ❌ Validation failed on attempt {attempt}")
        print(f"  Issues: {', '.join(feedback)}")
        return False, feedback

    def _check_post_reindex_status(self, plan):
        """Check if the function still needs fixing after reindexing."""
        if not self.reindex_occurred:
            return True  # No reindexing occurred, continue as normal
        
        print(f"   Checking if {plan.function_name} still needs fixing after reindexing...")
        
        # Re-search the codebase to see if the issue is resolved
        matches, updated_metadata = search_codebase(self.bug_description)
        
        if matches:
            # Check if our function is still flagged as needing fixes
            function_still_flagged = any(m['name'] == plan.function_name for m in matches)
            
            if not function_still_flagged:
                print(f"   {plan.function_name} appears to be resolved after reindexing")
                return False  # Don't need to fix this function
            else:
                print(f"  ⚠️ {plan.function_name} still needs fixing")
                return True
        else:
            print(f"   No functions flagged for fixing after reindexing")
            return False

    def _process_function_with_retry(self, plan):
        """Process a single function with enhanced retry logic."""
        print(f"Processing {plan.function_name} using {plan.strategy} strategy")
        
        # Check if we need to process this function post-reindex
        if self.post_reindex_validation and not self._check_post_reindex_status(plan):
            print(f"   Skipping {plan.function_name} - appears to be resolved")
            return True
        
        main_func = self._get_function_code(plan.function_name)
        if not main_func:
            print(f"❌ Could not find code for {plan.function_name}")
            return False

        helpers = [
            self._get_function_code(h) for h in plan.context_functions
            if self._get_function_code(h)
        ]

        # Initialize retry state
        attempt = 1
        feedback = []
        
        while attempt <= self.max_retries:
            print(f"  Attempt {attempt}/{self.max_retries}")
            
            # Build prompt (with retry feedback if this is a retry)
            if attempt == 1:
                prompt = self.planner.build_fix_prompt(
                    plan, main_func['code'], helpers, self.bug_description
                )
            else:
                prompt = self._build_retry_prompt(
                    plan, main_func['code'], helpers, attempt, feedback
                )

            # Generate fix
            fixed_code = generate_fix(
                instruction=prompt,
                main_function=main_func['code'],
                helper_functions=helpers
            )

            if not fixed_code:
                print(f"  ❌ No code generated on attempt {attempt}")
                attempt += 1
                continue

            # Validate and apply fix
            is_final = (attempt == self.max_retries)
            success, new_feedback = self._validate_and_apply_fix(
                plan, fixed_code, main_func, attempt, is_final
            )
            
            if success:
                return True
            
            feedback = new_feedback
            attempt += 1
        
        print(f"❌ Failed to fix {plan.function_name} after {self.max_retries} attempts")
        return False

    def run(self):
        print("Starting Enhanced Coordinator Agent with Advanced Retry Logic...")

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

        # Track success/failure across batches
        total_functions = sum(len(batch) for batch in fix_batches)
        successful_fixes = 0
        failed_fixes = 0

        for batch_idx, batch in enumerate(fix_batches, start=1):
            print(f"\n--- Executing Batch {batch_idx}/{len(fix_batches)} ---")

            batch_success = 0
            for plan in batch:
                if self._process_function_with_retry(plan):
                    successful_fixes += 1
                    batch_success += 1
                else:
                    failed_fixes += 1
            
            print(f"Batch {batch_idx} completed: {batch_success}/{len(batch)} functions fixed")
            
            # After each batch, check if reindexing occurred and if we should re-evaluate
            if self.reindex_occurred and batch_idx < len(fix_batches):
                print(f"\n Reindexing occurred. Re-evaluating remaining batches...")
                # You could implement logic here to re-plan remaining batches
                # For now, we'll continue with the current plan but check each function

        print(f"\n✅ Coordinator finished!")
        print(f" Summary: {successful_fixes}/{total_functions} functions fixed successfully")
        if failed_fixes > 0:
            print(f"⚠️ {failed_fixes} functions could not be fixed")
        
        if self.reindex_occurred:
            print(f" Codebase was reindexed during the process")
        
        # Final validation
        print("\n Running final validation...")
        matches, _ = search_codebase(self.bug_description)
        if not matches:
            print("✅ No remaining issues found in codebase!")
        else:
            remaining_issues = [m['name'] for m in matches]
            print(f"⚠️ Remaining issues in: {remaining_issues}")