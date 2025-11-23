#!/usr/bin/env python3
"""Test import speed of dependencies."""

import time
import sys

print("=" * 60)
print("Testing import speed...")
print("=" * 60)

# Test 1: Import torch
print("\n[1/3] Importing torch...", flush=True)
sys.stdout.flush()
t0 = time.time()
try:
    import torch
    t1 = time.time()
    print(f"[OK] torch imported in {t1-t0:.2f}s", flush=True)
    print(f"  - CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as e:
    print(f"[FAIL] Failed to import torch: {e}", flush=True)

# Test 2: Import sentence_transformers
print("\n[2/3] Importing sentence_transformers...", flush=True)
sys.stdout.flush()
t0 = time.time()
try:
    from sentence_transformers import SentenceTransformer
    t1 = time.time()
    print(f"[OK] sentence_transformers imported in {t1-t0:.2f}s", flush=True)
except Exception as e:
    print(f"[FAIL] Failed to import sentence_transformers: {e}", flush=True)

# Test 3: Load BGE model
print("\n[3/3] Loading BGE model (this will take 10-30s)...", flush=True)
sys.stdout.flush()
t0 = time.time()
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  - Using device: {device}", flush=True)

    model = SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device=device,
        local_files_only=True
    )
    t1 = time.time()
    print(f"[OK] BGE model loaded in {t1-t0:.2f}s", flush=True)

    # Quick test
    print("\n[BONUS] Testing embedding generation...", flush=True)
    t0 = time.time()
    embedding = model.encode(["Hello world"], normalize_embeddings=True)
    t1 = time.time()
    print(f"[OK] Generated embedding in {t1-t0:.3f}s", flush=True)
    print(f"  - Embedding shape: {embedding.shape}", flush=True)

except Exception as e:
    print(f"[FAIL] Failed to load BGE model: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
