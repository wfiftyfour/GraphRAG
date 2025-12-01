#!/usr/bin/env python3
"""Check if Traditional RAG system is properly set up."""

import sys
from pathlib import Path

# Color codes for terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_mark(passed):
    """Return check or cross mark."""
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def print_section(title):
    """Print section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    required = (3, 8)
    passed = version >= required

    print(f"{check_mark(passed)} Python Version: {version.major}.{version.minor}.{version.micro}")
    if not passed:
        print(f"   {YELLOW}Warning: Python 3.8+ required{RESET}")
    return passed


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")

    packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('yaml', 'pyyaml'),
        ('sentence_transformers', 'sentence-transformers'),
        ('torch', 'torch'),
        ('faiss', 'faiss-cpu'),
        ('gradio', 'gradio'),
        ('requests', 'requests'),
    ]

    all_passed = True
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"{check_mark(True)} {package_name}")
        except ImportError:
            print(f"{check_mark(False)} {package_name} - Not installed")
            all_passed = False

    return all_passed


def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")

    base_dir = Path(__file__).parent

    required_files = [
        'configs/rag_config.yaml',
        'src/indexing/chunker.py',
        'src/indexing/embedder.py',
        'src/indexing/vector_store.py',
        'src/retrieval/retriever.py',
        'src/generation/llm_client.py',
        'src/evaluation/metrics.py',
        'scripts/build_index.py',
        'scripts/evaluate.py',
        'app.py',
        'requirements.txt',
    ]

    all_passed = True
    for file_path in required_files:
        full_path = base_dir / file_path
        passed = full_path.exists()
        print(f"{check_mark(passed)} {file_path}")
        if not passed:
            all_passed = False

    return all_passed


def check_data():
    """Check if data files exist."""
    print("\nChecking data files...")

    base_dir = Path(__file__).parent

    # Check source data from GraphRAG
    graphrag_data = base_dir.parent / 'graphrag-system' / 'data' / 'input' / 'graphrag_format.jsonl'
    passed1 = graphrag_data.exists()
    print(f"{check_mark(passed1)} GraphRAG data: {graphrag_data}")
    if not passed1:
        print(f"   {YELLOW}Warning: Source data not found. Need GraphRAG data first.{RESET}")

    # Check if index is built
    index_file = base_dir / 'data' / 'processed' / 'embeddings' / 'faiss_index.bin'
    passed2 = index_file.exists()
    print(f"{check_mark(passed2)} FAISS index: {index_file}")
    if not passed2:
        print(f"   {YELLOW}Info: Run 'python scripts/build_index.py' to build index{RESET}")

    chunks_file = base_dir / 'data' / 'processed' / 'chunks' / 'chunks.json'
    passed3 = chunks_file.exists()
    print(f"{check_mark(passed3)} Chunks file: {chunks_file}")

    return passed1  # Only source data is critical


def check_ollama():
    """Check if Ollama is running."""
    print("\nChecking Ollama...")

    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print(f"{check_mark(True)} Ollama is running")

            # Check if model exists
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            has_model = any('qwen2.5:3b' in m for m in models)
            print(f"{check_mark(has_model)} qwen2.5:3b model installed")
            if not has_model:
                print(f"   {YELLOW}Run: ollama pull qwen2.5:3b{RESET}")
            return has_model
        else:
            print(f"{check_mark(False)} Ollama connection failed")
            return False
    except Exception as e:
        print(f"{check_mark(False)} Ollama not running or not accessible")
        print(f"   {YELLOW}Start Ollama: ollama serve{RESET}")
        return False


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")

    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"{check_mark(True)} CUDA available: {device}")
            return True
        else:
            print(f"{check_mark(False)} CUDA not available (will use CPU)")
            print(f"   {YELLOW}Info: GPU will speed up embedding generation{RESET}")
            return False
    except:
        print(f"{check_mark(False)} Could not check GPU")
        return False


def main():
    """Run all checks."""
    print_section("Traditional RAG System - Setup Check")

    checks = []

    # Critical checks
    print_section("Critical Requirements")
    checks.append(('Python Version', check_python_version()))
    checks.append(('Dependencies', check_dependencies()))
    checks.append(('Required Files', check_files()))
    checks.append(('Data Files', check_data()))

    # Optional checks
    print_section("Optional Components")
    checks.append(('Ollama', check_ollama()))
    checks.append(('GPU', check_gpu()))

    # Summary
    print_section("Summary")

    critical_checks = checks[:4]
    optional_checks = checks[4:]

    critical_passed = sum(1 for _, passed in critical_checks if passed)
    optional_passed = sum(1 for _, passed in optional_checks if passed)

    print(f"\nCritical: {critical_passed}/{len(critical_checks)} passed")
    print(f"Optional: {optional_passed}/{len(optional_checks)} passed")

    if critical_passed == len(critical_checks):
        print(f"\n{GREEN}✓ System is ready!{RESET}")
        print("\nNext steps:")
        print("  1. Build index: python scripts/build_index.py")
        print("  2. Run app: python app.py")
        return 0
    else:
        print(f"\n{RED}✗ System is not ready{RESET}")
        print("\nPlease fix the issues above before proceeding.")
        print("See QUICKSTART.md for setup instructions.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
