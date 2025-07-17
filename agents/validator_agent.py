import ast
import subprocess

def check_syntax(code: str) -> bool:
    """Check Python syntax using ast.parse."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")
        return False

def run_mypy(file_path: str) -> bool:
    """Run mypy type checks on the file."""
    result = subprocess.run(
        ["mypy", file_path],
        capture_output=True,
        text=True
    )
    print("[mypy output]:")
    print(result.stdout)
    return result.returncode == 0

def run_black_check(file_path: str) -> bool:
    """Check formatting using black --check."""
    result = subprocess.run(
        ["black", "--check", file_path],
        capture_output=True,
        text=True
    )
    print("[black output]:")
    print(result.stdout)
    return result.returncode == 0

def run_pytest() -> dict:
    """Run pytest and detect if tests were actually run."""
    result = subprocess.run(
        ["pytest"],
        capture_output=True,
        text=True
    )
    print("[pytest output]:")
    print(result.stdout)

    if "collected 0 items" in result.stdout.lower():
        return {"status": "no_tests", "success": False}

    return {"status": "ran_tests", "success": result.returncode == 0}

def validate_fix(code: str, file_path: str) -> dict:
    """
    Validate the fix with syntax, mypy, black, pytest.
    """
    report = {"syntax_check": False, "mypy": False, "black": False, "pytest": False,
              "pytest_status": "unknown", "all_pass": False}

    # Syntax first
    syntax_ok = check_syntax(code)
    report["syntax_check"] = syntax_ok

    if not syntax_ok:
        print("Skipping other checks due to syntax error.")
        return report

    # Static checks
    report["mypy"] = run_mypy(file_path)
    report["black"] = run_black_check(file_path)

    # Runtime test
    pytest_result = run_pytest()
    report["pytest"] = pytest_result["success"]
    report["pytest_status"] = pytest_result["status"]

    # Pass = syntax + mypy + black + tests ran & passed
    report["all_pass"] = report["syntax_check"] and report["mypy"] and report["black"] and report["pytest"]

    return report
