"""
vector_store.py — Vedology RAG Pipeline
Step 3: Build the vector index + document store from embeddings

What this does:
  - Loads embeddings.npy + chunk_ids.json (from embedder.py)
  - Loads chunks.jsonl (for the full metadata)
  - Builds a FAISS index for fast similarity search
  - Builds a dict mapping chunk_id → full metadata
  - Saves both to disk

Requirements:
    pip install faiss-cpu numpy

Run:
    python vector_store.py

Output:
    vedology.index        — FAISS binary index file
    vedology.docstore.json — id → full chunk metadata

Also produces VectorStore class used directly by retriever.py
"""

import json
import numpy as np
import faiss
from pathlib import Path


EMBEDDINGS_FILE  = "C:/Users/hetbh/Desktop/embeddings.npy"
IDS_FILE         = "C:/Users/hetbh/Desktop/chunk_ids.json"
CHUNKS_FILE      = "C:/Users/hetbh/Desktop/PBL-1/cleaning/chunks.jsonl"
INDEX_FILE       = "C:/Users/hetbh/Desktop/vedology.index"
DOCSTORE_FILE    = "C:/Users/hetbh/Desktop/vedology.docstore.json"


class VectorStore:
    """
    Wraps FAISS index + document store into one clean interface.
    This is what retriever.py imports and calls.

    Usage:
        store = VectorStore.load()
        results = store.search(query_vector, top_k=5)
        # results = [{'score': 0.91, 'chunk': {...full metadata...}}, ...]
    """

    def __init__(self, index: faiss.Index, docstore: dict, ids: list[str]):
        self.index = index
        self.docstore = docstore
        self.ids = ids

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Find the top_k most similar chunks to query_vector.

        query_vector must be:
          - shape (1, 1024) — 2D, one row
          - dtype float32
          - L2 normalised (same as how you embedded chunks)

        Returns list of dicts, highest score first:
          [{'score': 0.91, 'id': 'chanakya_dharm_v11', 'chunk': {...}}]
        """
        # Ensure correct shape and dtype
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)

        # FAISS search — returns distances and row indices
        scores, row_indices = self.index.search(query_vector, top_k)

        results = []
        for score, row_idx in zip(scores[0], row_indices[0]):
            if row_idx == -1:
                # FAISS returns -1 when fewer results exist than top_k
                continue
            chunk_id = self.ids[row_idx]
            chunk = self.docstore.get(chunk_id)
            if chunk:
                results.append({
                    "score": float(score),
                    "id": chunk_id,
                    "chunk": chunk,
                })

        return results

    def search_with_filter(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        source_filter: str | None = None,
        entry_type_filter: str | None = None,
    ) -> list[dict]:
        """
        Search with optional pre-filters on source or entry_type.

        Example:
            # Only return Chanakya results
            results = store.search_with_filter(vec, source_filter="Chanakya Neeti")

            # Only return Garuda verses
            results = store.search_with_filter(vec, entry_type_filter="Verse")

        Strategy: retrieve more candidates then filter.
        We fetch top_k * 5 candidates and filter down.
        Crude but correct — at 940 chunks this is plenty.
        """
        candidates = self.search(query_vector, top_k=min(top_k * 5, len(self.ids)))

        filtered = []
        for r in candidates:
            meta = r["chunk"].get("metadata", r["chunk"])
            if source_filter and meta.get("source") != source_filter:
                continue
            if entry_type_filter and meta.get("entry_type") != entry_type_filter:
                continue
            filtered.append(r)
            if len(filtered) == top_k:
                break

        return filtered

    @classmethod
    def load(cls, index_file: str = INDEX_FILE, docstore_file: str = DOCSTORE_FILE,
             ids_file: str = IDS_FILE) -> "VectorStore":
        """Load an already-built store from disk. Used by retriever.py."""
        index = faiss.read_index(index_file)

        with open(docstore_file, encoding="utf-8") as f:
            docstore = json.load(f)

        with open(ids_file) as f:
            ids = json.load(f)

        print(f"VectorStore loaded: {index.ntotal} vectors, {len(docstore)} documents")
        return cls(index, docstore, ids)


def build(
    embeddings_file: str = EMBEDDINGS_FILE,
    ids_file: str = IDS_FILE,
    chunks_file: str = CHUNKS_FILE,
    index_file: str = INDEX_FILE,
    docstore_file: str = DOCSTORE_FILE,
) -> VectorStore:
    """
    Build the FAISS index and document store from scratch.
    Run this once after embedder.py produces embeddings.npy.
    """
    print("=== Vedology Vector Store Builder ===\n")

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file).astype(np.float32)
    print(f"  Shape: {embeddings.shape}  (chunks × dimensions)")
    print(f"  dtype: {embeddings.dtype}")

    with open(ids_file) as f:
        ids = json.load(f)
    print(f"  IDs loaded: {len(ids)}\n")

    assert len(ids) == embeddings.shape[0], (
        f"Mismatch: {len(ids)} IDs but {embeddings.shape[0]} embedding rows"
    )

    # ── Build FAISS index ─────────────────────────────────────────────────────
    dim = embeddings.shape[1]

    # IndexFlatIP = exact inner product search
    # "IP" = inner product = cosine similarity when vectors are L2-normalised
    # "Flat" = brute force — checks every vector
    # At 940 chunks this is instantaneous. Switch to IndexHNSWFlat at ~100k chunks.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built:")
    print(f"  Type: IndexFlatIP (exact cosine similarity)")
    print(f"  Vectors indexed: {index.ntotal}")
    print(f"  Dimensions: {dim}\n")

    # ── Build document store ───────────────────────────────────────────────────
    # Simple dict: chunk_id → full chunk object
    # This is what gets returned to the LLM after retrieval
    print(f"Building document store from {chunks_file}...")
    docstore = {}
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line.strip())
            docstore[chunk["id"]] = chunk
    print(f"  Documents stored: {len(docstore)}\n")

    # ── Save to disk ───────────────────────────────────────────────────────────
    faiss.write_index(index, index_file)
    with open(docstore_file, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False)

    print(f"Saved:")
    print(f"  {index_file}      — FAISS binary index")
    print(f"  {docstore_file}   — document store")

    return VectorStore(index, docstore, ids)


def smoke_test(store: VectorStore, embeddings: np.ndarray, ids: list[str]):
    """
    Quick sanity check: retrieve chunk 0 using its own vector.
    Should return itself as the top result with score ~1.0
    """
    print("\n=== Smoke Test ===")
    print("Querying with the vector of chunk 0 — should return itself first...")

    results = store.search(embeddings[0], top_k=3)

    for i, r in enumerate(results):
        print(f"  Rank {i+1}: score={r['score']:.4f}  id={r['id']}")

    top = results[0]
    if top["id"] == ids[0]:
        print(f"\n  PASS — top result is the query chunk itself (score={top['score']:.4f})")
    else:
        print(f"\n  WARN — top result is {top['id']}, expected {ids[0]}")
        print("  This is unusual. Check that embeddings are L2 normalised.")


def main():
    # Check inputs exist
    for f in [EMBEDDINGS_FILE, IDS_FILE, CHUNKS_FILE]:
        if not Path(f).exists():
            print(f"ERROR: {f} not found.")
            print("Run embedder.py first to generate embeddings.npy and chunk_ids.json")
            return

    store = build()

    # Run smoke test
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    with open(IDS_FILE) as f:
        ids = json.load(f)
    smoke_test(store, embeddings, ids)

    print("\n=== SUMMARY ===")
    print(f"Vector store ready.")
    print(f"To use in retriever.py:")
    print(f"  from vector_store import VectorStore")
    print(f"  store = VectorStore.load()")
    print(f"  results = store.search(query_vector, top_k=5)")


if __name__ == "__main__":
    main()
