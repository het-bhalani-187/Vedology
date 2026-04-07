"""
embedder.py — FIXED & ROBUST VERSION

Reads:   JSONL chunks
Outputs: embeddings.npy + chunk_ids.json

Works with BOTH formats:
1. {"chunk_text": "..."}
2. {"content": {"original": "..."}}
"""

import json
import time
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------
CHUNKS_FILE = "C:/Users/hetbh/Desktop/PBL-1/cleaning/VidhurNiti_chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "chunk_ids.json"
MODEL_NAME = "intfloat/multilingual-e5-large"

PASSAGE_PREFIX = "passage: "


# ---------- LOAD CHUNKS (SAFE) ----------
def load_chunks(filepath: str):
    chunks = []

    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
                chunks.append(data)
            except json.JSONDecodeError as e:
                print(f"❌ Skipping bad JSON at line {i}")
                print(f"   {line[:100]}")
                continue

    print(f"✅ Loaded {len(chunks)} valid chunks")
    return chunks


# ---------- EXTRACT TEXT ----------
def extract_text(chunk: dict):
    """
    Supports multiple formats safely
    """

    # Format 1: {"chunk_text": "..."}
    if "chunk_text" in chunk:
        return chunk["chunk_text"]

    # Format 2: {"content": {"original": "..."}}
    if "content" in chunk and isinstance(chunk["content"], dict):
        return chunk["content"].get("original", "")

    # Fallback
    return ""


# ---------- EMBEDDING ----------
def embed_chunks(chunks, model):
    texts = []
    ids = []

    for c in chunks:
        text = extract_text(c).strip()

        if not text:
            continue

        texts.append(PASSAGE_PREFIX + text)
        ids.append(c.get("id", f"chunk_{len(ids)}"))

    print(f"\n📦 Embedding {len(texts)} chunks...\n")

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print(f"\n✅ Done in {time.time() - start:.2f}s")

    return embeddings, ids


# ---------- MAIN ----------
def main():
    print("\n=== Vedology Embedder (Fixed) ===\n")

    # Check file
    if not Path(CHUNKS_FILE).exists():
        print(f"❌ File not found: {CHUNKS_FILE}")
        return

    # Load model
    print(f"📥 Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Model loaded ({model.get_sentence_embedding_dimension()} dims)\n")

    # Load chunks
    chunks = load_chunks(CHUNKS_FILE)

    if not chunks:
        print("❌ No valid chunks found. Fix your JSONL file.")
        return

    # Embed
    embeddings, ids = embed_chunks(chunks, model)

    # Save
    np.save(EMBEDDINGS_FILE, embeddings)

    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Chunks embedded:   {len(ids)}")
    print(f"Vector shape:      {embeddings.shape}")
    print(f"Saved:")
    print(f"  → {EMBEDDINGS_FILE}")
    print(f"  → {IDS_FILE}")


# ---------- RUN ----------
if __name__ == "__main__":
    main()
