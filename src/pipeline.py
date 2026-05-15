"""Stage 1 query pipeline: question → retrieve → prompt → LLM → answer.

This is the online half of RAG. It reads the persisted Chroma index built
by `python -m src.ingest.build_index`, then assembles an LCEL chain that
takes a question and returns an answer grounded in retrieved chunks.

Usage:
    python -m src.pipeline "Who won the 2024 Monaco Grand Prix?"
"""
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHROMA_DIR,
    COLLECTION_NAME,
    LLM_MODEL,
    TOP_K,
)


# The prompt template. Two placeholders: {context} (the retrieved chunks)
# and {question} (what the user asked). The instructions tell the model
# to answer ONLY from context and to say so when it can't.
PROMPT_TEMPLATE = """You are an F1 expert assistant. Answer the question using ONLY the context provided below. If the answer is not in the context, say "I don't have enough information to answer that." Do not make up facts.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list) -> str:
    """Turn a list of retrieved Documents into a single string for the prompt.

    Each chunk is separated by a blank line and prefixed with its source race,
    which both helps the LLM use the chunks and gives us cheap citations.
    """
    formatted = []
    for d in docs:
        race = d.metadata.get("race", "unknown")
        formatted.append(f"[Source: {race}]\n{d.page_content}")
    return "\n\n".join(formatted)


def build_chain():
    """Wire up the full RAG chain using LCEL."""
    # 1. Load the persisted vector store. Note: we pass the SAME embedding
    #    model used during ingest. Embedding models are not interchangeable —
    #    vectors from one model are meaningless to another.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # 2. .as_retriever() turns the vector store into a Runnable. It takes a
    #    query string and returns a list of Documents (top-k by similarity).
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # 3. The prompt template — also a Runnable. When invoked with a dict
    #    {"context": ..., "question": ...} it returns formatted messages.
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 4. The LLM — also a Runnable. Takes messages, returns an AIMessage.
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # 5. The output parser — extracts the .content string from the AIMessage.
    parser = StrOutputParser()

    # 6. The chain itself. Read this top-to-bottom:
    #
    #    The dict at the top says: build a {context, question} dict where:
    #      - context comes from the retriever's output, then formatted to string
    #      - question is the original input, passed through unchanged
    #    Then pipe that dict into the prompt, then into the LLM, then into the parser.
    #
    #    RunnablePassthrough() means "whatever input came in, just pass it along."
    #    The retriever takes a string and returns docs, so we chain it with format_docs.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    return chain, retriever


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.pipeline "your question here"')
        sys.exit(1)
    question = " ".join(sys.argv[1:])

    chain, retriever = build_chain()

    # For learning: show what was retrieved BEFORE we generate. Seeing the
    # raw retrieval is how you debug RAG — if these chunks are wrong, the
    # answer will be wrong no matter how good the LLM is.
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    print("\nRetrieved chunks:")
    retrieved = retriever.invoke(question)
    for i, doc in enumerate(retrieved, 1):
        race = doc.metadata.get("race", "?")
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"\n  [{i}] ({race})")
        print(f"      {snippet}...")

    print("\n" + "=" * 60)
    print("Generating answer...")
    print("=" * 60)
    answer = chain.invoke(question)
    print(f"\n{answer}\n")


if __name__ == "__main__":
    main()