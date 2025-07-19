# utils/reindex.py

import os
import time
from scripts.embed_function import extract_functions_from_folder, embed_and_store

def reindex_codebase(folder_path: str, verbose: bool = True):
    """
    Run full codebase re-extraction and re-embedding with detailed progress.
    
    Args:
        folder_path: Path to the codebase folder
        verbose: Whether to show detailed progress
    """
    if verbose:
        print(f"  📁 Scanning {folder_path} for functions...")
    
    start_time = time.time()
    
    # Extract functions
    functions = extract_functions_from_folder(folder_path)
    
    if verbose:
        print(f"  📊 Found {len(functions)} functions to reindex")
        
        # Show which files are being processed
        files_found = set()
        for func in functions:
            if 'file_path' in func:
                files_found.add(func['file_path'])
        
        if files_found:
            print(f"  📄 Processing {len(files_found)} files:")
            for file_path in sorted(files_found):
                # Show relative path for cleaner output
                rel_path = os.path.relpath(file_path, folder_path) if os.path.isabs(file_path) else file_path
                print(f"    - {rel_path}")
    
    # Embed and store
    if verbose:
        print(f"  🔄 Embedding and storing functions...")
    
    embed_and_store(functions)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if verbose:
        print(f"  ✅ Reindexing completed in {duration:.2f} seconds")
        print(f"  📊 Total functions reindexed: {len(functions)}")
    
    return {
        'functions_count': len(functions),
        'files_processed': len(files_found) if 'files_found' in locals() else 0,
        'duration': duration
    }

def reindex_codebase_silent(folder_path: str):
    """Silent version of reindex_codebase for when you don't want output."""
    return reindex_codebase(folder_path, verbose=False)