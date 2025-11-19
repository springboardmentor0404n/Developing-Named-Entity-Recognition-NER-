# build_index.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm import tqdm

DATA_CSV = "financeinsight_labeled_with_positive.csv"  # adjust if needed
OUTPUT_DIR = "./index_data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # compact, fast

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_CSV)
# use the cleaned text column you use (here: clean_text). Adjust if needed.
if "clean_text" in df.columns:
    texts = df["clean_text"].dropna().astype(str).tolist()
else:
    texts = df["text"].dropna().astype(str).tolist()

print(f"Loaded {len(texts)} texts.")

# Optional: chunk long texts into smaller passages (naive split on sentences)
def chunk_text(text, max_sentences=4):
    sents = text.split(". ")
    chunks = []
    for i in range(0, len(sents), max_sentences):
        chunk = ". ".join(sents[i:i+max_sentences]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks

print("Chunking texts...")
all_chunks = []
meta_rows = []
for idx, t in enumerate(tqdm(texts, desc="texts")):
    chunks = chunk_text(t)
    for c_i, c in enumerate(chunks):
        all_chunks.append(c)
        meta_rows.append({"orig_idx": idx, "chunk_id": f"{idx}_{c_i}", "text": c})

print(f"Total chunks: {len(all_chunks)}")

print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

batch_size = 64
embeddings = []
for i in tqdm(range(0, len(all_chunks), batch_size), desc="embed"):
    batch = all_chunks[i:i+batch_size]
    embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    embeddings.append(embs)
embeddings = np.vstack(embeddings).astype("float32")
d = embeddings.shape[1]
print("Embeddings shape:", embeddings.shape)

print("Building FAISS index (IndexFlatIP + normalize for cosine)...")
# normalize vectors for cosine-sim (use InnerProduct on normalized vectors)
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(d)
index.add(embeddings)
print("Index size:", index.ntotal)

# Save index
index_path = os.path.join(OUTPUT_DIR, "faiss_index.idx")
faiss.write_index(index, index_path)
print("Saved FAISS index ->", index_path)

# Save metadata
meta_df = pd.DataFrame(meta_rows)
meta_path = os.path.join(OUTPUT_DIR, "index_metadata.csv")
meta_df.to_csv(meta_path, index=False)
print("Saved metadata ->", meta_path)

# Save model name/params
with open(os.path.join(OUTPUT_DIR, "index_info.pkl"), "wb") as f:
    pickle.dump({"model": EMBED_MODEL, "dim": d}, f)

print("Done.")
