import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import subprocess
from utils.patch_function import patch_function_in_file
from utils.embedding import search_codebase
from graphs.call_analysis import get_called_functions, get_all_called_functions, detect_globals
from agents.planner_agent import build_multi_function_prompt
from agents.validator_agent import validate_fix
from agents.fixer_agent import generate_fix
import ast
import re
import json



# Run the model via Ollama
with open("config.json", "r") as f:
    CONFIG = json.load(f)

OLLAMA_MODEL = CONFIG["OLLAMA_MODEL"]

def ask_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode()







# Main execution
if __name__ == "__main__":
    print(" AutoFixer — Local Bug Fixer (CodeLlama via Ollama)")

    issue = input("\n Enter the bug description:\n> ")
    matches,metadata = search_codebase(issue)

    if not matches:
        print(" No matching function found.")
        exit()

    match = matches[0]
    print("\n Matched Function:\n")
    print(match['code'])

    called_funcs = get_all_called_functions(match['code'], metadata)

    helper_matches = []
    for name in called_funcs:
        for func in metadata:
            if func['name'] == name:
                helper_matches.append(func)

    print(f"\n Matched Helper Functions: {[f['name'] for f in helper_matches]}")

    globals_in_func = detect_globals(match['code'])
    globals_context = ""
    for var in globals_in_func:
        globals_context += f"{var} = <provide_value_here>\n"

    print(f"\n Detected Globals: {globals_in_func}")


    instruction = f"Fix {match['name']} to resolve: {issue}"
    MAX_TRIES = 3
    for attempt in range(MAX_TRIES):
        print(f"\nAttempt {attempt + 1}:")

        raw_output = generate_fix(
            instruction=instruction,
            main_function=match['code'],
            helper_functions=helper_matches
        ).strip()

        for wrapper in ["'''", '"""', '```']:
            if raw_output.startswith(wrapper) and raw_output.endswith(wrapper):
                raw_output = raw_output[len(wrapper):-len(wrapper)].strip()

        print("\nSuggested Fix:\n")
        print(raw_output)
        print("\n" + "-" * 60)

        fixed_blocks = re.findall(r'(def [\s\S]*?)(?=\ndef |\Z)', raw_output.strip())
        cleaned_blocks = []

        for block in fixed_blocks:
            block = block.strip()
            if block.endswith('"""') or block.endswith("'''"):
                block = block.rsplit('"""', 1)[0].rsplit("'''", 1)[0].strip()
            cleaned_blocks.append(block)

        print(f"\nDetected {len(cleaned_blocks)} updated function(s). Patching...\n")

        all_pass = True
        all_candidates = [match] + helper_matches

        for fixed_code in cleaned_blocks:
            for func in all_candidates:
                if func['name'] in fixed_code:
                    func['issue'] = issue
                    patch_function_in_file(func, fixed_code)

                    validation_result = validate_fix(fixed_code, func['file'])
                    if validation_result["all_pass"]:
                        print(f"Fix for {func['name']} passed validation.")
                    else:
                        print(f"Validation failed for {func['name']}.")
                        print(validation_result)
                        all_pass = False

                        # Add validator errors to next instruction
                        error_feedback = "\n".join(
                            f"{k}: {v}" for k, v in validation_result.items() if k != "all_pass"
                        )
                        instruction += f"\nValidator feedback:\n{error_feedback}\nPlease fix these issues."

        if all_pass:
            print("\nAll fixes passed validation.")
            break
        else:
            print("\nRetrying with validator feedback...")

    else:
        print("\nReached maximum attempts. Some fixes may still fail.")