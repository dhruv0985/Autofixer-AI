import subprocess
import json
import re
from typing import List, Dict, Optional

# Load config
with open("config.json", "r") as f:
    CONFIG = json.load(f)

MODEL = "gemma3"  # Local Ollama model

def ask_ollama(prompt: str, max_retries: int = 3) -> str:
    """Run Ollama with the given prompt and return raw output with retry logic."""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["ollama", "run", MODEL],
                input=prompt,
                text=True,
                encoding='utf-8',
                capture_output=True,
                check=True,
                timeout=60  # 60 second timeout
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Ollama error (attempt {attempt + 1}): {e.stderr}")
            if attempt == max_retries - 1:
                return ""
        except subprocess.TimeoutExpired:
            print(f"Ollama timeout (attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return ""
    return ""

def extract_python_code(text: str) -> str:
    """Extract valid Python code from LLM output with improved parsing."""
    if not text.strip():
        return ""
    
    # First, try to find code blocks with various markers
    code_block_patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```(.*?)```',
        r"'''python\n(.*?)'''",
        r"'''\n(.*?)'''",
        r"'''(.*?)'''",
        r'"""python\n(.*?)"""',
        r'"""\n(.*?)"""',
        r'"""(.*?)"""'
    ]

    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code and _is_valid_python_structure(code):
                return code

    # If no code blocks found, try to extract function definitions
    lines = text.split('\n')
    code_lines = []
    in_function = False
    indent_level = 0
    
    for line in lines:
        # Skip obvious non-code lines
        if any(marker in line.lower() for marker in [
            'explanation:', 'note:', 'here is', 'here\'s', 'the following',
            'this function', 'this code', 'output:', 'result:'
        ]):
            if in_function:
                break
            continue
            
        # Check if this is a function definition
        if line.strip().startswith('def '):
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            code_lines.append(line)
        elif in_function:
            current_indent = len(line) - len(line.lstrip())
            # If we're back to the same or less indentation and it's not empty, we might be done
            if line.strip() and current_indent <= indent_level and not line.strip().startswith('#'):
                if line.strip().startswith('def '):
                    # Another function definition
                    code_lines.append(line)
                    indent_level = current_indent
                else:
                    # End of function
                    break
            else:
                code_lines.append(line)
        elif line.strip().startswith(('import ', 'from ')):
            code_lines.append(line)

    if code_lines:
        extracted = '\n'.join(code_lines).strip()
        if _is_valid_python_structure(extracted):
            return extracted

    # Last resort: clean up the entire text
    cleaned = text.strip()
    
    # Remove common wrapper patterns
    for wrapper in ["'''", '"""', '```']:
        if cleaned.startswith(wrapper) and cleaned.endswith(wrapper):
            cleaned = cleaned[len(wrapper):-len(wrapper)].strip()
    
    # Remove explanatory prefixes
    cleaned = re.sub(r'^(Here is|Here\'s|The following|Below is|Updated|Fixed).*?:\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'```\w*\s*', '', cleaned)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    
    return cleaned.strip()

def _is_valid_python_structure(code: str) -> bool:
    """Check if the code has valid Python structure."""
    if not code.strip():
        return False
    
    # Must contain at least one function definition
    if 'def ' not in code:
        return False
    
    # Should not contain obvious non-code patterns
    invalid_patterns = [
        'explanation:', 'note:', 'here is the', 'output:', 'result:',
        'this function', 'this code', '**explanation**', '**note**'
    ]
    
    code_lower = code.lower()
    for pattern in invalid_patterns:
        if pattern in code_lower:
            return False
    
    return True

def generate_fix(instruction: str, main_function: str, helper_functions: List[Dict]) -> str:
    """
    Generate a fix using Ollama with improved prompt engineering.
    
    Args:
        instruction: The fix instruction (can be a full prompt from planner)
        main_function: The main function code to fix
        helper_functions: List of helper function dicts with 'name' and 'code'
    
    Returns:
        Fixed code as string
    """
    
    # Check if instruction is already a full prompt (from planner)
    if "You are an expert Python" in instruction or "INSTRUCTIONS:" in instruction:
        # This is a full prompt from the planner
        prompt = instruction
    else:
        # Build a simple prompt for backward compatibility
        helpers_text = "\n\n".join([
            f"# Helper function: {h['name']}\n{h['code']}" 
            for h in helper_functions
        ]) if helper_functions else ""
        
        prompt = f"""
You are an expert Python developer.

Fix the following function according to the instruction.

# Function to fix:
{main_function}

# Helper functions (for context):
{helpers_text}

# Fix instruction:
{instruction}

RULES:
- Return ONLY the corrected Python function(s)
- Keep original function signatures
- Make minimal changes
- No explanations or comments
- No code fences or markdown
- Each function should start with 'def'
""".strip()

    # Generate fix
    raw_output = ask_ollama(prompt)
    
    if not raw_output:
        print("No output from Ollama")
        return ""
    
    # Extract and clean code
    clean_code = extract_python_code(raw_output)
    
    if not clean_code:
        print("Could not extract valid code from Ollama output")
        print(f"Raw output: {raw_output[:200]}...")
        return ""
    
    return clean_code

def validate_generated_code(code: str, original_function: str) -> bool:
    """
    Validate that generated code is reasonable.
    
    Args:
        code: Generated code
        original_function: Original function code
    
    Returns:
        True if code seems valid
    """
    if not code.strip():
        return False
    
    # Check for basic Python syntax
    try:
        import ast
        ast.parse(code)
    except SyntaxError:
        return False
    
    # Check that it contains function definitions
    if 'def ' not in code:
        return False
    
    # Check that it's not just the original function
    if code.strip() == original_function.strip():
        return False
    
    return True

# Backward compatibility function
def generate_simple_fix(instruction: str, main_function: str, helper_functions: List[Dict]) -> str:
    """Simple fix generation for backward compatibility."""
    return generate_fix(instruction, main_function, helper_functions)