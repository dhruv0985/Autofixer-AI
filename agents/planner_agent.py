import networkx as nx
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class FixStrategy(Enum):
    ISOLATED = "isolated"  # Fix function in isolation
    DEPENDENT = "dependent"  # Fix function considering its dependencies
    BATCH = "batch"  # Fix multiple related functions together
    CASCADING = "cascading"  # Fix in dependency order

@dataclass
class FixPlan:
    function_name: str
    strategy: FixStrategy
    dependencies: List[str]
    dependents: List[str]
    priority: float
    context_functions: List[str]
    batch_group: int = 0
    semantic_relevance: float = 0.0  # New field for semantic matching

class PlannerAgent:
    """
    Intelligent planner that analyzes function dependencies and creates
    an optimal fix sequence considering call relationships, data flow,
    and semantic relevance to the bug description.
    """
    
    def __init__(self, call_graph: nx.DiGraph, data_flow_graph: nx.DiGraph, 
                 semantic_graph: nx.DiGraph, unified_graph: nx.DiGraph):
        self.call_graph = call_graph
        self.data_flow_graph = data_flow_graph
        self.semantic_graph = semantic_graph
        self.unified_graph = unified_graph
        
    def calculate_semantic_relevance(self, func_name: str, bug_description: str) -> float:
        """
        Calculate how semantically relevant a function is to the bug description.
        Higher score means more relevant.
        """
        if not bug_description:
            return 0.0
        
        bug_lower = bug_description.lower()
        func_lower = func_name.lower()
        
        # Direct function name mention gets highest priority
        if func_lower in bug_lower:
            return 10.0
        
        # Partial matches for function names
        if any(part in bug_lower for part in func_lower.split('_')):
            return 5.0
        
        # Keyword matching based on common bug patterns
        relevance_keywords = {
            'calculate': ['calculate', 'computation', 'math', 'formula'],
            'process': ['process', 'handle', 'execute'],
            'validate': ['validate', 'check', 'verify'],
            'parse': ['parse', 'read', 'extract'],
            'transform': ['transform', 'convert', 'change'],
            'step': ['step', 'phase', 'stage'],
            'main': ['main', 'entry', 'start'],
            'helper': ['helper', 'utility', 'support']
        }
        
        keyword_score = 0.0
        for category, keywords in relevance_keywords.items():
            if any(keyword in bug_lower for keyword in keywords):
                if category in func_lower:
                    keyword_score += 2.0
        
        return keyword_score
        
    def analyze_function_dependencies(self, target_functions: List[str]) -> Dict[str, Dict]:
        """Analyze dependencies for each target function."""
        dependency_analysis = {}
        
        for func in target_functions:
            if func not in self.call_graph:
                continue
                
            # Direct dependencies (functions this function calls)
            direct_deps = list(self.call_graph.successors(func))
            
            # Reverse dependencies (functions that call this function)
            reverse_deps = list(self.call_graph.predecessors(func))
            
            # Data flow dependencies
            data_deps = []
            if func in self.data_flow_graph:
                data_deps = list(self.data_flow_graph.successors(func))
            
            # Calculate dependency depth
            try:
                paths_to_main = list(nx.all_simple_paths(self.call_graph, func, 'main'))
                depth = min(len(path) for path in paths_to_main) if paths_to_main else 0
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                depth = 0
            
            dependency_analysis[func] = {
                'direct_dependencies': direct_deps,
                'reverse_dependencies': reverse_deps,
                'data_dependencies': data_deps,
                'depth_from_main': depth,
                'is_leaf': len(direct_deps) == 0,
                'is_root': len(reverse_deps) == 0
            }
            
        return dependency_analysis
    
    def calculate_fix_priority(self, func: str, analysis: Dict, bug_description: str = "") -> float:
        """Calculate priority score for fixing a function with semantic awareness."""
        func_analysis = analysis[func]
        
        # Semantic relevance (highest weight)
        semantic_score = self.calculate_semantic_relevance(func, bug_description)
        
        # Traditional dependency-based factors
        depth_factor = 1.0 / (func_analysis['depth_from_main'] + 1)
        dependency_factor = len(func_analysis['direct_dependencies']) * 0.1
        impact_factor = len(func_analysis['reverse_dependencies']) * 0.2
        leaf_bonus = 0.5 if func_analysis['is_leaf'] else 0.0
        
        # Combine scores with semantic relevance having the highest weight
        if semantic_score > 0:
            # If semantically relevant, prioritize it heavily
            priority = semantic_score * 2.0 + depth_factor + dependency_factor + impact_factor + leaf_bonus
        else:
            # Fall back to traditional dependency-based priority
            priority = depth_factor + dependency_factor + impact_factor + leaf_bonus
        
        return priority
    
    def determine_fix_strategy(self, func: str, analysis: Dict, target_functions: List[str]) -> FixStrategy:
        """Determine the best strategy for fixing a function."""
        func_analysis = analysis[func]
        
        # If function has no dependencies, can fix in isolation
        if func_analysis['is_leaf']:
            return FixStrategy.ISOLATED
        
        # If all dependencies are also being fixed, use batch strategy
        deps_being_fixed = [dep for dep in func_analysis['direct_dependencies'] 
                           if dep in target_functions]
        
        if len(deps_being_fixed) == len(func_analysis['direct_dependencies']):
            return FixStrategy.BATCH
        
        # If some dependencies are being fixed, use cascading
        if deps_being_fixed:
            return FixStrategy.CASCADING
        
        # Default to dependent strategy
        return FixStrategy.DEPENDENT
    
    def create_fix_batches(self, fix_plans: List[FixPlan]) -> List[List[FixPlan]]:
        """
        Group fix plans into batches with semantic relevance considered.
        Functions with high semantic relevance get their own batch first.
        """
        batches = []
        remaining_plans = fix_plans.copy()
        
        # First, check if any function has very high semantic relevance (direct mention)
        high_priority_plans = [p for p in remaining_plans if p.semantic_relevance >= 10.0]
        
        if high_priority_plans:
            # Create a priority batch for directly mentioned functions
            batches.append(high_priority_plans)
            for plan in high_priority_plans:
                remaining_plans.remove(plan)
        
        # Then create dependency-based batches for the rest
        while remaining_plans:
            current_batch = []
            
            # Find plans that can be executed in this batch
            for plan in remaining_plans[:]:
                # Check if all dependencies are already fixed
                deps_satisfied = all(
                    dep_name not in [p.function_name for p in remaining_plans]
                    for dep_name in plan.dependencies
                )
                
                if deps_satisfied:
                    current_batch.append(plan)
                    remaining_plans.remove(plan)
            
            # If no plans can be executed, break to avoid infinite loop
            if not current_batch:
                # Add remaining plans anyway (might be circular dependencies)
                current_batch = remaining_plans
                remaining_plans = []
            
            batches.append(current_batch)
        
        return batches
    
    def get_context_functions(self, target_func: str, strategy: FixStrategy) -> List[str]:
        """Get helper functions that should be included as context."""
        context = []
        
        if strategy == FixStrategy.ISOLATED:
            # No additional context needed
            return context
        
        elif strategy == FixStrategy.DEPENDENT:
            # Include direct dependencies
            if target_func in self.call_graph:
                context.extend(self.call_graph.successors(target_func))
        
        elif strategy == FixStrategy.BATCH:
            # Include all related functions in the batch
            if target_func in self.call_graph:
                context.extend(self.call_graph.successors(target_func))
                context.extend(self.call_graph.predecessors(target_func))
        
        elif strategy == FixStrategy.CASCADING:
            # Include dependencies and their dependencies
            if target_func in self.call_graph:
                # Get dependencies up to 2 levels deep
                deps_level1 = list(self.call_graph.successors(target_func))
                context.extend(deps_level1)
                
                for dep in deps_level1:
                    if dep in self.call_graph:
                        context.extend(self.call_graph.successors(dep))
        
        return list(set(context))  # Remove duplicates
    
    def create_fix_plan(self, target_functions: List[str], bug_description: str) -> List[List[FixPlan]]:
        """Create a comprehensive fix plan for the target functions."""
        print(f"Creating fix plan for functions: {target_functions}")
        print(f"Bug description: {bug_description}")
        
        # Analyze dependencies
        dependency_analysis = self.analyze_function_dependencies(target_functions)
        
        # Create individual fix plans
        fix_plans = []
        for func in target_functions:
            if func not in dependency_analysis:
                continue
                
            priority = self.calculate_fix_priority(func, dependency_analysis, bug_description)
            strategy = self.determine_fix_strategy(func, dependency_analysis, target_functions)
            context_funcs = self.get_context_functions(func, strategy)
            semantic_relevance = self.calculate_semantic_relevance(func, bug_description)
            
            plan = FixPlan(
                function_name=func,
                strategy=strategy,
                dependencies=dependency_analysis[func]['direct_dependencies'],
                dependents=dependency_analysis[func]['reverse_dependencies'],
                priority=priority,
                context_functions=context_funcs,
                semantic_relevance=semantic_relevance
            )
            fix_plans.append(plan)
        
        # Sort by priority (higher priority first)
        fix_plans.sort(key=lambda x: x.priority, reverse=True)
        
        # Create batches with semantic awareness
        batches = self.create_fix_batches(fix_plans)
        
        # Log the plan
        print(f"Created {len(batches)} fix batches:")
        for i, batch in enumerate(batches):
            print(f"  Batch {i + 1}: {[plan.function_name for plan in batch]}")
            for plan in batch:
                print(f"    - {plan.function_name}: {plan.strategy.value}, priority={plan.priority:.3f}, semantic={plan.semantic_relevance:.1f}")
        
        return batches
    
    def build_fix_prompt(self, plan: FixPlan, main_func_code: str, 
                        helper_funcs: List[Dict], bug_description: str) -> str:
        """Build a targeted prompt based on the fix strategy."""
        
        base_prompt = f"""
You are an expert Python code fixer.

Bug Description: {bug_description}

Target Function to Fix: {plan.function_name}
Fix Strategy: {plan.strategy.value}
Semantic Relevance: {plan.semantic_relevance:.1f}

Main Function:
{main_func_code}
"""
        
        if helper_funcs:
            base_prompt += "\nHelper Functions (for context):\n"
            for helper in helper_funcs:
                base_prompt += f"{helper['code']}\n"
        
        # Add strategy-specific instructions
        if plan.strategy == FixStrategy.ISOLATED:
            base_prompt += """
INSTRUCTIONS:
- Fix ONLY the main function above
- Make minimal changes to resolve the bug
- Keep the function signature unchanged
- Return ONLY the corrected function code
"""
        
        elif plan.strategy == FixStrategy.DEPENDENT:
            base_prompt += """
INSTRUCTIONS:
- Fix the main function considering its dependencies (helper functions)
- If helper functions need changes to fix the bug, update them too
- Make minimal changes to resolve the bug
- Return ALL updated functions (main + any modified helpers)
"""
        
        elif plan.strategy == FixStrategy.BATCH:
            base_prompt += """
INSTRUCTIONS:
- Fix the main function and its related functions as a cohesive unit
- Ensure all functions work together correctly after the fix
- Make minimal changes to resolve the bug
- Return ALL updated functions
"""
        
        elif plan.strategy == FixStrategy.CASCADING:
            base_prompt += """
INSTRUCTIONS:
- Fix the main function considering the cascade effect on dependent functions
- Update any helper functions that need changes
- Ensure the fix doesn't break the dependency chain
- Return ALL updated functions
"""
        
        base_prompt += """
RULES:
- Return ONLY valid Python code
- No explanations, comments, or markdown
- Each function should start with 'def'
- Preserve original function signatures
"""
        
        return base_prompt.strip()