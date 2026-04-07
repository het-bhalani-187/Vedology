"""
retriever.py — Vedology RAG Pipeline
Step 4: Take a user query → return the most relevant chunks

This is the core of the RAG system.
Everything before this was setup. This runs on every user query.

What this does:
  1. Loads the vector store (index + docstore)
  2. Loads the embedding model
  3. Embeds the user query (with "query: " prefix)
  4. Searches FAISS for top-k nearest chunks
  5. Filters by score threshold (drops irrelevant results)
  6. Returns ranked list of chunks ready for the LLM

Requirements:
    pip install sentence-transformers faiss-cpu numpy

Usage (as a module):
    from retriever import Retriever
    retriever = Retriever()  # loads model + index once
    results = retriever.retrieve("what does Chanakya say about karma?")

Usage (standalone test):
    python retriever.py
"""

import json
import time
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore


MODEL_NAME       = "intfloat/multilingual-e5-large"
QUERY_PREFIX     = "query: "       # required for multilingual-e5
DEFAULT_TOP_K    = 5               # how many chunks to retrieve
SCORE_THRESHOLD  = 0.70            # drop chunks below this — they're noise
                                   # tune this after testing: too high = misses,
                                   # too low = irrelevant results


@dataclass
class RetrievalResult:
    """One retrieved chunk with its score and full metadata."""
    id: str
    score: float
    source: str
    entry_type: str
    chunk_text: str
    metadata: dict

    def display(self):
        """Human-readable summary for debugging."""
        meta = self.metadata
        if self.entry_type == "Shloka":
            title = f"Chanakya Neeti | {meta.get('section')} | Shloka {meta.get('shloka_number')}"
        else:
            title = f"Garuda Purana | {meta.get('chapter_number_display')} | Verse {meta.get('verse_number')}"

        print(f"  [{self.score:.3f}] {title}")
        meaning = meta.get("meaning") or meta.get("verse_text", "")
        if meaning:
            print(f"          {meaning[:120]}...")


class Retriever:
    """
    The retrieval engine. Instantiate once, call retrieve() many times.

    Loading the model takes ~3 seconds.
    After that, each query takes ~100-150ms (dominated by embedding time).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        print("Initialising Retriever...")

        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.store = VectorStore.load()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.dim = self.model.get_sentence_embedding_dimension()

        print(f"Ready in {time.time()-t0:.1f}s\n")

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.
        Returns shape (1, dim), dtype float32, L2 normalised.
        """
        prefixed = QUERY_PREFIX + query.strip()
        vector = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.astype(np.float32)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        source_filter: str | None = None,
        entry_type_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Main retrieval method. Call this with the user's raw query string.

        Args:
            query:              Raw user query — no preprocessing needed
            top_k:              Override default number of results
            source_filter:      "Chanakya Neeti" or "Garuda Purana (Saroddhara)"
            entry_type_filter:  "Shloka" or "Verse"

        Returns:
            List of RetrievalResult, sorted by score descending.
            Only includes results above score_threshold.
        """
        k = top_k or self.top_k

        t0 = time.time()

        # Embed the query
        query_vector = self._embed_query(query)

        # Search
        if source_filter or entry_type_filter:
            raw_results = self.store.search_with_filter(
                query_vector,
                top_k=k,
                source_filter=source_filter,
                entry_type_filter=entry_type_filter,
            )
        else:
            raw_results = self.store.search(query_vector, top_k=k)

        # Build result objects, apply threshold
        results = []
        for r in raw_results:
            if r["score"] < self.score_threshold:
                continue

            chunk = r["chunk"]
            meta = chunk.get("metadata", chunk)

            results.append(RetrievalResult(
                id=r["id"],
                score=r["score"],
                source=chunk.get("source", ""),
                entry_type=chunk.get("entry_type", ""),
                chunk_text=chunk.get("chunk_text", ""),
                metadata=meta,
            ))

        elapsed_ms = (time.time() - t0) * 1000
        return results, elapsed_ms

    def retrieve_for_llm(self, query: str, **kwargs) -> str:
        """
        Convenience method: retrieve and format results as a
        single context string ready to be injected into an LLM prompt.

        This is what Step 5 (LLM Processing) will call.
        """
        results, _ = self.retrieve(query, **kwargs)

        if not results:
            return "No relevant passages found in the Vedology knowledge base."

        parts = []
        for i, r in enumerate(results, 1):
            meta = r.metadata
            if r.entry_type == "Shloka":
                header = (
                    f"[{i}] Chanakya Neeti — {meta.get('section')} — "
                    f"Shloka {meta.get('shloka_number')} "
                    f"(similarity: {r.score:.2f})"
                )
                body = meta.get("meaning", "")
                sanskrit = meta.get("sanskrit", "")
                if sanskrit:
                    body = f"Sanskrit: {sanskrit}\n{body}"
            else:
                header = (
                    f"[{i}] Garuda Purana — {meta.get('chapter_title')} — "
                    f"Verse {meta.get('verse_number')} "
                    f"(similarity: {r.score:.2f})"
                )
                body = meta.get("verse_text", "")

            keywords = meta.get("keywords", [])
            if keywords:
                body += f"\nKeywords: {', '.join(keywords[:5])}"

            parts.append(f"{header}\n{body}")

        return "\n\n---\n\n".join(parts)


def run_test_queries(retriever: Retriever):
    """
    Run a set of test queries to validate retrieval quality.
    Read these results carefully — they tell you if your pipeline works.
    """
    test_queries = [
        # English conceptual
        "what does Chanakya say about karma and consequences?",
        # English thematic
        "importance of education and knowledge",
        # Sanskrit transliteration
        "Chalaa Laxmishchalaah meaning",
        # Garuda Purana specific
        "what happens to the soul after death?",
        # Cross-source
        "truth and righteousness",
        # Short casual query
        "best friend",
        # Edge case — nonsense, should return low scores or nothing
        "how to make pizza",
    ]

    print("=" * 60)
    print("RETRIEVAL TEST QUERIES")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        results, elapsed_ms = retriever.retrieve(query)

        if not results:
            print(f"  No results above threshold. ({elapsed_ms:.0f}ms)")
        else:
            print(f"  {len(results)} results in {elapsed_ms:.0f}ms:")
            for r in results:
                r.display()

    print("\n" + "=" * 60)
    print("Test complete.")
    print()
    print("What to look for:")
    print("  karma query    → should return Dharm section shlokas 14, 15, 17")
    print("  education      → should return Knowledge/Students section")
    print("  transliteration→ should still find shloka 11 (Chalaa = dharma)")
    print("  death/soul     → should return Garuda Purana chapters I/II/XVI")
    print("  pizza          → should return nothing (below threshold)")


def main():
    retriever = Retriever()
    run_test_queries(retriever)

    # Also show the formatted context string for one query
    print("\n=== Context string for LLM (karma query) ===\n")
    context = retriever.retrieve_for_llm("what does Chanakya say about karma?")
    print(context)


if __name__ == "__main__":
    main()
