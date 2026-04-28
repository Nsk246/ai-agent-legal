import os
import json
import time
import numpy as np
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, MAX_MEMORY_RESULTS

_embedder = None
_vector_store = None


def get_embedder() -> HuggingFaceEmbeddings:
    """Return cached embedder — loads once, reused forever."""
    global _embedder
    if _embedder is None:
        print("[MEMORY] Loading embedding model (one-time)...")
        _embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[MEMORY] Embedding model ready.")
    return _embedder


def prewarm():
    """Call this at server startup to avoid cold start during first request."""
    get_embedder()
    _get_store()
    print("[MEMORY] Vector store pre-warmed.")


def _get_store() -> FAISS | None:
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    index_file = f"{FAISS_INDEX_PATH}.faiss"
    if os.path.exists(index_file):
        try:
            _vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                get_embedder(),
                allow_dangerous_deserialization=True,
            )
            print(f"[MEMORY] Loaded FAISS index from {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"[MEMORY WARNING] Could not load index: {e}")
    return _vector_store


def _save_store():
    global _vector_store
    if _vector_store:
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
        _vector_store.save_local(FAISS_INDEX_PATH)


@tool
def store_movie_analysis(movie_title: str, analysis_summary: str) -> str:
    """
    Store a completed movie analysis in long-term vector memory (FAISS).
    Use after finishing analysis to persist findings for future retrieval.
    Input: movie_title and analysis_summary as strings.
    """
    global _vector_store
    if not movie_title or not analysis_summary:
        return "[MEMORY ERROR] Both movie_title and analysis_summary are required."
    try:
        doc = Document(
            page_content=analysis_summary,
            metadata={
                "movie":     movie_title,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type":      "analysis",
            },
        )
        store = _get_store()
        if store is None:
            _vector_store = FAISS.from_documents([doc], get_embedder())
        else:
            store.add_documents([doc])
            _vector_store = store
        _save_store()
        return f"[MEMORY] Analysis for '{movie_title}' stored successfully."
    except Exception as e:
        return f"[MEMORY ERROR] Failed to store: {str(e)}"


@tool
def retrieve_similar_analyses(query: str) -> str:
    """
    Retrieve past movie analyses from long-term memory similar to the query.
    Use to find previously analyzed films or related plot hole patterns.
    Input: a search query string.
    """
    store = _get_store()
    if store is None:
        return "[MEMORY] No past analyses found. This is the first analysis."
    if not query or not query.strip():
        return "[MEMORY ERROR] Empty query provided."
    try:
        results = store.similarity_search_with_score(query, k=MAX_MEMORY_RESULTS)
        if not results:
            return "[MEMORY] No similar past analyses found."
        formatted = []
        for doc, score in results:
            similarity_pct = max(0, round((1 - score) * 100, 1))
            formatted.append(
                f"Movie: {doc.metadata.get('movie', 'Unknown')}\n"
                f"Date: {doc.metadata.get('timestamp', 'Unknown')}\n"
                f"Similarity: {similarity_pct}%\n"
                f"Summary: {doc.page_content[:400]}"
            )
        return "Past analyses:\n\n" + "\n---\n".join(formatted)
    except Exception as e:
        return f"[MEMORY ERROR] Retrieval failed: {str(e)}"


@tool
def store_user_preference(preference_key: str, preference_value: str) -> str:
    """
    Store a user preference in long-term memory.
    Examples: genre tolerance, favorite directors, spoiler preferences.
    Input: preference_key and preference_value as strings.
    """
    global _vector_store
    if not preference_key or not preference_value:
        return "[MEMORY ERROR] Both key and value required."
    try:
        content = f"User preference — {preference_key}: {preference_value}"
        doc = Document(
            page_content=content,
            metadata={
                "type":      "user_preference",
                "key":       preference_key,
                "value":     preference_value,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        store = _get_store()
        if store is None:
            _vector_store = FAISS.from_documents([doc], get_embedder())
        else:
            store.add_documents([doc])
            _vector_store = store
        _save_store()
        return f"[MEMORY] Preference '{preference_key}: {preference_value}' saved."
    except Exception as e:
        return f"[MEMORY ERROR] Failed to store preference: {str(e)}"
