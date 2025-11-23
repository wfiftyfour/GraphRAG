#!/usr/bin/env python3
"""Check BGE model download status."""

from pathlib import Path
import os

# HuggingFace cache directory (Windows)
cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
model_dir = cache_dir / 'models--BAAI--bge-large-en-v1.5'

print(f"Cache directory: {cache_dir}")
print(f"Model directory: {model_dir}")
print(f"Model directory exists: {model_dir.exists()}")

if model_dir.exists():
    # Calculate total size of downloaded files
    total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
    size_gb = total_size / (1024**3)

    print(f"\nTotal size downloaded: {size_gb:.2f} GB")
    print(f"Expected size: ~1.34 GB")

    if size_gb > 1.2:
        print("\n[OK] Download complete! Model is ready to use.")
    elif size_gb > 0.1:
        completion = (size_gb / 1.34) * 100
        print(f"\n[IN PROGRESS] Download in progress: {completion:.1f}% complete")
    else:
        print("\n[STARTING] Download just started...")
else:
    print("\n[NOT FOUND] Model directory not found. Download has not started yet.")
    print("\nTo start download, the app needs to initialize and load the embedder.")
