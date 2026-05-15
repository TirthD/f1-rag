"""Stage 1 ingestion: Wikipedia → chunks → embeddings → Chroma.

Run this ONCE to build the vector store. After it runs, you'll have a
persistent chroma_db/ folder that the query pipeline reads from.

Usage:
    python -m src.ingest.build_index
"""
import os
import time

# Set a polite User-Agent BEFORE importing WebBaseLoader. Wikipedia's API
# rejects generic Python user agents — they require identifying info.
os.environ["USER_AGENT"] = (
    "f1-rag-learning-project/0.1 (https://github.com/TirthD/f1-rag; contact via github)"
)

from langchain_community.document_loaders import WebBaseLoader  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402

from src.config import (  # noqa: E402
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# We hit Wikipedia directly via URL. Spaces in titles become "_" on Wikipedia.
WIKI_BASE = "https://en.wikipedia.org/wiki/"

RACES_2024 = [
    "2024 Bahrain Grand Prix",
    "2024 Saudi Arabian Grand Prix",
    "2024 Australian Grand Prix",
    "2024 Japanese Grand Prix",
    "2024 Chinese Grand Prix",
    "2024 Miami Grand Prix",
    "2024 Emilia Romagna Grand Prix",
    "2024 Monaco Grand Prix",
    "2024 Canadian Grand Prix",
    "2024 Spanish Grand Prix",
    "2024 Austrian Grand Prix",
    "2024 British Grand Prix",
    "2024 Hungarian Grand Prix",
    "2024 Belgian Grand Prix",
    "2024 Dutch Grand Prix",
    "2024 Italian Grand Prix",
    "2024 Azerbaijan Grand Prix",
    "2024 Singapore Grand Prix",
    "2024 United States Grand Prix",
    "2024 Mexico City Grand Prix",
    "2024 São Paulo Grand Prix",
    "2024 Las Vegas Grand Prix",
    "2024 Qatar Grand Prix",
    "2024 Abu Dhabi Grand Prix",
]


def title_to_url(title: str) -> str:
    """'2024 Monaco Grand Prix' -> 'https://en.wikipedia.org/wiki/2024_Monaco_Grand_Prix'"""
    return WIKI_BASE + title.replace(" ", "_")


def load_documents() -> list:
    """Fetch each race's Wikipedia page. Returns a list of Documents."""
    docs = []
    for title in RACES_2024:
        url = title_to_url(title)
        print(f"  Loading: {title}")
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            # Attach our own clean metadata. The default 'source' is the URL;
            # we add the race title so we can filter/cite by it later.
            for d in loaded:
                d.metadata["race"] = title
                d.metadata["season"] = 2024
            docs.extend(loaded)
        except Exception as e:
            print(f"    ⚠ failed for {title}: {e}")
        # Be polite — small delay between requests so we don't hammer Wikipedia.
        time.sleep(0.5)
    return docs


def chunk_documents(docs: list) -> list:
    """Split each Document into smaller, retrieval-sized chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_vector_store(chunks: list) -> Chroma:
    """Embed chunks and persist them to disk in Chroma."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def main():
    print(f"Step 1/3: loading {len(RACES_2024)} Wikipedia pages...")
    docs = load_documents()
    print(f"  → got {len(docs)} documents\n")

    if not docs:
        print("No documents loaded. Aborting.")
        return

    print(f"Step 2/3: chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(docs)
    print(f"  → produced {len(chunks)} chunks\n")

    print(f"Step 3/3: embedding and writing to Chroma at {CHROMA_DIR}...")
    vs = build_vector_store(chunks)
    print(f"  → done. Collection '{COLLECTION_NAME}' contains "
          f"{vs._collection.count()} vectors.\n")

    print("Index built. You can now run the query pipeline.")


if __name__ == "__main__":
    main()