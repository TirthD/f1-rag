"""Smoke test: verifies that LangChain, Ollama, and HuggingFace embeddings all work.

Run after `pip install -r requirements.txt` and after pulling an Ollama model.

Usage:
    python smoke_test.py
"""
import sys


def test_ollama():
    print("→ Testing Ollama connection...")
    try:
        from langchain_ollama import ChatOllama
        from src.config import LLM_MODEL
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("    Did you install requirements.txt and activate your venv?")
        return False

    try:
        llm = ChatOllama(model=LLM_MODEL)
        response = llm.invoke("Reply with exactly: PONG")
        print(f"  ✓ Ollama responded: {response.content!r}")
        return True
    except Exception as e:
        print(f"  ✗ Ollama call failed: {e}")
        print(f"    Is Ollama running? Is the model '{LLM_MODEL}' pulled?")
        print(f"    Try: ollama pull {LLM_MODEL}")
        return False


def test_embeddings():
    print("→ Testing embeddings...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from src.config import EMBEDDING_MODEL
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False

    try:
        emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vec = emb.embed_query("Lewis Hamilton")
        print(f"  ✓ Embedding model loaded. Vector dimension: {len(vec)}")
        return True
    except Exception as e:
        print(f"  ✗ Embedding call failed: {e}")
        return False


def test_chroma():
    print("→ Testing Chroma...")
    try:
        from langchain_chroma import Chroma  # noqa: F401
        print("  ✓ Chroma import works.")
        return True
    except ImportError as e:
        print(f"  ✗ Chroma import failed: {e}")
        return False


if __name__ == "__main__":
    results = [test_ollama(), test_embeddings(), test_chroma()]
    print()
    if all(results):
        print("All checks passed. You're ready for Stage 1.")
        sys.exit(0)
    else:
        print("Some checks failed. Fix them before continuing.")
        sys.exit(1)
