# F1 RAG

An end-to-end RAG (Retrieval-Augmented Generation) pipeline over Formula 1 data, built for learning the core concepts and the LangChain / LangGraph ecosystem.

## Stack

- **Embeddings:** `all-MiniLM-L6-v2` via `langchain-huggingface` (local, free, CPU-friendly)
- **Vector store:** Chroma (local, file-backed)
- **LLM:** Ollama (local, free) — default model `llama3.1:8b`
- **Orchestration:** LangChain (Stages 1–2), LangGraph (Stage 3)
- **Evaluation:** Custom eval set + optional LangSmith / Ragas later

## One-time setup

### 1. Install Ollama and pull a model

Ollama runs open-source LLMs locally. Install it from <https://ollama.com/> (one-click installer for macOS / Windows / Linux). Once installed, pull a model:

```bash
ollama pull llama3.1:8b
```

If you have less than 8GB of RAM free, use a smaller model instead:

```bash
ollama pull mistral:7b      # ~4GB
# or
ollama pull llama3.2:3b     # ~2GB, lower quality but works on anything
```

Verify it works:

```bash
ollama run llama3.1:8b "Say hello"
```

(You can `Ctrl-D` to exit the interactive prompt.) Ollama runs as a background service on `http://localhost:11434`; LangChain will talk to it through that.

### 2. Set up the Python environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# or on Windows:
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify the install

```bash
python -c "from langchain_ollama import ChatOllama; print(ChatOllama(model='llama3.1:8b').invoke('hi').content)"
```

If that prints a greeting, everything is wired up.

## Project layout

```
f1-rag/
├── data/                     # raw + processed data (gitignored)
├── src/
│   ├── ingest/               # data loaders, chunkers, embedder
│   ├── retrieve/             # vector / BM25 / hybrid / rerank
│   ├── sql/                  # parameterized SQL templates (Stage 3)
│   ├── config.py             # model names, paths, constants
│   ├── router.py             # query routing (Stage 3)
│   ├── generate.py           # LLM call + prompt template
│   └── pipeline.py           # ties everything together
├── eval/
│   ├── questions.json        # eval set
│   └── run_eval.py
├── notebooks/                # for exploration and one-offs
├── requirements.txt
├── .gitignore
└── README.md
```

## Stages

The project is built incrementally — each stage teaches specific concepts.

- **Stage 1 — Naive RAG.** Load Wikipedia pages → chunk → embed → store in Chroma → simple LCEL chain (`retriever | prompt | llm | parser`). Goal: see the whole pipeline run end-to-end and feel the limitations.
- **Stage 2 — Better retrieval.** Metadata-aware chunking, hybrid BM25 + vector search via `EnsembleRetriever`, cross-encoder reranking, `SelfQueryRetriever` for natural-language metadata filters. Build the eval set.
- **Stage 3 — Structured + unstructured fusion.** Add Jolpica F1 API data into SQLite, build a LangGraph router that decides between SQL lookup, retrieval, or both.
- **Stage 4 — Evaluation.** Formalize with LangSmith and/or Ragas. Ablation studies. Document what worked.

## Running

(Each stage adds entry points. For now, the scaffold has placeholders.)

```bash
python -m src.pipeline "Who won the 2024 Monaco Grand Prix?"
```
