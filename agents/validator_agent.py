import ast
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple

def check_syntax(code: str) -> Tuple[bool, str]:
    """Check Python syntax using ast.parse with detailed error info."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        return False, error_msg

def run_mypy(file_path: str) -> Tuple[bool, str]:
    """Run mypy type checks on the file with detailed output."""
    result = subprocess.run(
        ["mypy", file_path],
        capture_output=True,
        text=True
    )
    
    success = result.returncode == 0
    output = result.stdout.strip()
    
    if not success and result.stderr:
        output += "\n" + result.stderr.strip()
    
    print("[mypy output]:")
    print(output)
    
    return success, output

def run_black_check(file_path: str) -> Tuple[bool, str]:
    """Check formatting using black --check with detailed output."""
    result = subprocess.run(
        ["black", "--check", file_path],
        capture_output=True,
        text=True
    )
    
    success = result.returncode == 0
    output = result.stdout.strip()
    
    if not success and result.stderr:
        output += "\n" + result.stderr.strip()
    
    print("[black output]:")
    print(output if output else "No formatting issues found")
    
    return success, output

def auto_format_with_black(file_path: str) -> Tuple[bool, str]:
    """Automatically format the file with black and return the result."""
    result = subprocess.run(
        ["black", file_path],
        capture_output=True,
        text=True
    )
    
    success = result.returncode == 0
    output = result.stdout.strip()
    
    if result.stderr:
        output += "\n" + result.stderr.strip()
    
    print("[black auto-format]:")
    print("Code automatically formatted" if success else f"Formatting failed: {output}")
    
    return success, output

def run_pytest() -> Dict[str, any]:
    """Run pytest with detailed status information."""
    result = subprocess.run(
        ["pytest", "-v"],
        capture_output=True,
        text=True
    )
    
    output = result.stdout.strip()
    print("[pytest output]:")
    print(output)
    
    # Parse pytest output for more details
    if "collected 0 items" in output.lower():
        return {"status": "no_tests", "success": False, "output": output}
    
    # Extract test results
    lines = output.split('\n')
    passed = sum(1 for line in lines if '::' in line and 'PASSED' in line)
    failed = sum(1 for line in lines if '::' in line and 'FAILED' in line)
    
    return {
        "status": "ran_tests",
        "success": result.returncode == 0,
        "output": output,
        "passed": passed,
        "failed": failed
    }

def validate_code_structure(code: str, original_code: str) -> Dict[str, any]:
    """Validate that the code structure is reasonable."""
    issues = []
    
    # Check if code is not empty
    if not code.strip():
        issues.append("Generated code is empty")
        return {"valid": False, "issues": issues}
    
    # Check for function definitions
    if 'def ' not in code:
        issues.append("No function definitions found")
    
    # Check if code is not identical to original
    if code.strip() == original_code.strip():
        issues.append("Generated code is identical to original")
    
    # Check for obvious placeholders
    placeholder_patterns = ["TODO", "FIXME", "...", "pass  # TODO"]
    for pattern in placeholder_patterns:
        if pattern in code:
            issues.append(f"Found placeholder: {pattern}")
    
    # Try to parse AST for more detailed validation
    try:
        tree = ast.parse(code)
        
        # Check for function definitions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not functions:
            issues.append("No function definitions found in AST")
        
        # Check for return statements
        returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        if not returns:
            issues.append("No return statements found")
        
    except SyntaxError as e:
        issues.append(f"AST parsing failed: {e}")
    
    return {"valid": len(issues) == 0, "issues": issues}

def create_temp_file_with_code(code: str, original_file_path: str) -> str:
    """Create a temporary file with the new code for validation."""
    # Get the original file extension
    _, ext = os.path.splitext(original_file_path)
    if not ext:
        ext = '.py'
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
        f.write(code)
        return f.name

def validate_fix_enhanced(code: str, file_path: str, original_code: str = None, 
                         auto_format: bool = True) -> Dict[str, any]:
    """
    Enhanced validation with smart black formatting and detailed feedback.
    
    Args:
        code: The code to validate
        file_path: Original file path for context
        original_code: Original code for comparison
        auto_format: Whether to automatically format with black
    """
    report = {
        "syntax_check": False,
        "mypy": False,
        "black": False,
        "pytest": False,
        "structure_check": False,
        "pytest_status": "unknown",
        "all_pass": False,
        "detailed_feedback": {},
        "suggestions": [],
        "formatted_code": None  # Will contain auto-formatted code if applicable
    }
    
    # Structure validation (if original code provided)
    if original_code:
        structure_result = validate_code_structure(code, original_code)
        report["structure_check"] = structure_result["valid"]
        report["detailed_feedback"]["structure"] = structure_result["issues"]
        
        if not structure_result["valid"]:
            report["suggestions"].extend([
                "Ensure the code contains proper function definitions",
                "Make sure the code is not identical to the original",
                "Remove any placeholder text or incomplete implementations"
            ])
    
    # Syntax validation
    syntax_ok, syntax_msg = check_syntax(code)
    report["syntax_check"] = syntax_ok
    report["detailed_feedback"]["syntax"] = syntax_msg
    
    if not syntax_ok:
        report["suggestions"].append("Fix syntax errors before proceeding")
        print("Skipping other checks due to syntax error.")
        return report
    
    # Create temporary file for external tool validation
    temp_file = None
    try:
        temp_file = create_temp_file_with_code(code, file_path)
        
        # MyPy validation
        mypy_ok, mypy_msg = run_mypy(temp_file)
        report["mypy"] = mypy_ok
        report["detailed_feedback"]["mypy"] = mypy_msg
        
        if not mypy_ok:
            report["suggestions"].append("Fix type checking issues reported by mypy")
        
        # Black formatting validation with auto-fix option
        black_ok, black_msg = run_black_check(temp_file)
        report["black"] = black_ok
        report["detailed_feedback"]["black"] = black_msg
        
        if not black_ok and auto_format:
            print("  🔧 Attempting to auto-format code...")
            format_ok, format_msg = auto_format_with_black(temp_file)
            
            if format_ok:
                # Read the formatted code
                with open(temp_file, 'r') as f:
                    formatted_code = f.read()
                
                report["formatted_code"] = formatted_code
                report["black"] = True  # Mark as passing since we auto-formatted
                report["detailed_feedback"]["black"] = "Auto-formatted successfully"
                print("  ✅ Code auto-formatted successfully")
            else:
                report["suggestions"].append("Fix code formatting issues manually")
                print("  ❌ Auto-formatting failed")
        elif not black_ok:
            report["suggestions"].append("Fix code formatting issues (run 'black' on the code)")
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
    
    # Pytest validation (run in original directory)
    pytest_result = run_pytest()
    report["pytest"] = pytest_result["success"]
    report["pytest_status"] = pytest_result["status"]
    report["detailed_feedback"]["pytest"] = pytest_result["output"]
    
    if pytest_result["status"] == "no_tests":
        report["suggestions"].append("Consider adding tests to verify the fix")
    elif not pytest_result["success"]:
        report["suggestions"].append("Fix issues causing test failures")
    
    # Overall success (use formatted code if available)
    structure_ok = report["structure_check"] if original_code else True
    report["all_pass"] = (
        structure_ok and 
        report["syntax_check"] and 
        report["mypy"] and 
        report["black"] and 
        report["pytest"]
    )
    
    return report

def validate_fix_smart(code: str, file_path: str, original_code: str = None) -> Dict[str, any]:
    """
    Smart validation that handles common issues automatically.
    
    This is the recommended validation function that:
    1. Checks syntax first
    2. Auto-formats with black if needed
    3. Validates with mypy
    4. Runs tests if available
    5. Provides the best possible code
    """
    result = validate_fix_enhanced(code, file_path, original_code, auto_format=True)
    
    # If we have formatted code and it passes all checks, use it
    if result["formatted_code"] and result["all_pass"]:
        print("   Using auto-formatted code")
        return result
    
    # Otherwise, use the original result
    return result

# Backward compatibility
def validate_fix(code: str, file_path: str) -> dict:
    """
    Backward compatible validation function with smart formatting.
    """
    enhanced_result = validate_fix_smart(code, file_path)
    
    # Convert to old format, but include formatted code if available
    result = {
        "syntax_check": enhanced_result["syntax_check"],
        "mypy": enhanced_result["mypy"],
        "black": enhanced_result["black"],
        "pytest": enhanced_result["pytest"],
        "pytest_status": enhanced_result["pytest_status"],
        "all_pass": enhanced_result["all_pass"]
    }
    
    # Add formatted code if available
    if enhanced_result["formatted_code"]:
        result["formatted_code"] = enhanced_result["formatted_code"]
    
    return result