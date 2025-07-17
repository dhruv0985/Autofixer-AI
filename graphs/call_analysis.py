import ast

def get_called_functions(code: str) -> list:
    called = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Handle both simple calls (foo()) and attribute calls (self.foo())
                if isinstance(node.func, ast.Name):
                    called.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called.add(node.func.attr)
    except Exception as e:
        print(f" AST parsing failed: {e}")
    return list(called)



def get_all_called_functions(code: str, metadata: list, seen=None) -> set:
    from .call_analysis import get_called_functions  # if needed, or use local

    if seen is None:
        seen = set()

    new_calls = get_called_functions(code)
    for call in new_calls:
        if call not in seen:
            seen.add(call)
            for func in metadata:
                if func['name'] == call:
                    get_all_called_functions(func['code'], metadata, seen)
    return seen



def detect_globals(code: str) -> list:

    globals_found = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    if node.id.isupper():
                        globals_found.append(node.id)
    except Exception as e:
        print(f" AST parsing failed: {e}")
    return list(set(globals_found))