# query_index.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import os

INDEX_DIR = "./index_data"
TOP_K = 5

# load
print("Loading index & metadata...")
index = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.idx"))
meta = pd.read_csv(os.path.join(INDEX_DIR, "index_metadata.csv"))
with open(os.path.join(INDEX_DIR, "index_info.pkl"), "rb") as f:
    info = pickle.load(f)

model = SentenceTransformer(info["model"])

def query(q, k=TOP_K):
    emb = model.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(emb)
    D, I = index.search(emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        row = meta.iloc[idx].to_dict()
        row["score"] = float(score)
        results.append(row)
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query_index.py \"your query\" [k]")
        sys.exit(1)
    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K
    hits = query(q, k)
    for i, h in enumerate(hits):
        print(f"\n=== Hit {i+1} (score={h['score']:.4f}) ===")
        print(h["text"])
