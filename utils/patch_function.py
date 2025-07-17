import ast
import os
import shutil
from difflib import unified_diff

def patch_function_in_file(metadata: dict, new_code: str, dry_run=True) -> None:
    file_path = metadata['file']

    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    source = "".join(lines)

    # Use AST to find actual boundary
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == metadata["name"]:
            start_line = node.lineno - 1
            end_line = max(
                child.lineno for child in ast.walk(node) if hasattr(child, "lineno")
            )
            break
    else:
        print(f" Could not find function `{metadata['name']}`.")
        return

    old_block = lines[start_line:end_line]
    new_block = new_code.strip().splitlines(keepends=True)
    if not new_block:
        print(" New code block is empty — nothing to patch.")
        return
    if not new_block[-1].endswith("\n"):
        new_block[-1] += "\n"

    # Show diff for dry run
    diff = unified_diff(
        old_block, new_block,
        fromfile="original",
        tofile="updated",
        lineterm=""
    )
    print("\n".join(diff))

    if dry_run:
        confirm = input("\n Apply patch? [Y/N]: ").strip().lower()
        if confirm != "y":
            print(" Patch cancelled.")
            return

    # Backup
    backup_path = f"{file_path}.bak"
    shutil.copyfile(file_path, backup_path)

    # Apply patch
    lines[start_line:end_line] = new_block
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # Log patch
    with open("fix_log.txt", "a", encoding='utf-8') as log:
        log.write(f"\n Patched {file_path} | Lines {start_line+1}–{end_line}\n")
        log.write(f"Issue: {metadata.get('issue', 'Unknown')}\n")
        log.write("Fixed Function:\n")
        log.write(new_code + "\n")
        log.write("—" * 60 + "\n")

    print(f" Patch applied and logged!")
