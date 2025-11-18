# fast_store_index.py
"""
Single-file fast ingestion + retrieval for Pinecone (new API).
Features:
- Load PDFs from Data/
- Split into chunks (langchain text splitter)
- Batch embed with sentence-transformers (GPU if available)
- Cache embeddings & stable IDs
- Upsert vectors + metadata (text/snippet/source/chunk_idx) to Pinecone
- DocSearch wrapper for similarity_search returning langchain Document objects
"""

import glob
import os
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from langchain_classic.schema import Document
# LangChain / PDF / splitting / schema
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Pinecone new API
from pinecone import Pinecone, ServerlessSpec
# Embedding & device detection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# ----------------- CONFIG -----------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "medibot")
DATA_DIR = os.environ.get("DATA_DIR", "Data/")
EMBED_CACHE_FILE = Path(os.environ.get("EMBED_CACHE_FILE", "embed_cache.pkl"))
ID_CACHE_FILE = Path(os.environ.get("ID_CACHE_FILE", "id_cache.pkl"))

DIMENSION = int(os.environ.get("EMBED_DIM", 384))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", 128))
UPSERT_BATCH = int(os.environ.get("UPSERT_BATCH", 200))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
MODEL_NAME = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ----------------- Pinecone client -----------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# helper to get list of existing index names (handles variations in client)
try:
    existing_indexes = pc.list_indexes().names()
except Exception:
    try:
        existing_indexes = pc.list_indexes()
    except Exception:
        existing_indexes = []

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )
_index = pc.Index(INDEX_NAME)


# ----------------- Utilities -----------------
def _generate_id() -> str:
    return uuid.uuid4().hex


def load_single_pdf(path: str) -> str:
    """Return concatenated text of all pages in PDF at path."""
    loader = PyMuPDFLoader(path)
    pages = loader.load()
    texts = [p.page_content for p in pages]
    return " ".join(texts)


def load_all_pdfs(folder: str = DATA_DIR, max_workers: int = None) -> List[Document]:
    paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not paths:
        return []
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(load_single_pdf, paths),
                       total=len(paths), desc="Loading PDFs"))
    docs: List[Document] = []
    for idx, (text, path) in enumerate(zip(results, paths)):
        meta = {"source": os.path.basename(
            path), "path": os.path.abspath(path), "doc_idx": idx}
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def split_documents_to_chunks(documents: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Document] = []
    for doc in tqdm(documents, desc="Splitting docs"):
        splitted = splitter.split_documents([doc])
        for i, s in enumerate(splitted):
            meta = dict(s.metadata) if s.metadata else {}
            meta.update({"chunk_idx": i})
            snippet = (
                s.page_content[:400] + "...") if len(s.page_content) > 400 else s.page_content
            meta["snippet"] = snippet
            chunks.append(Document(page_content=s.page_content, metadata=meta))
    return chunks


def get_sentence_transformer(model_name: str = MODEL_NAME, device: str = None) -> SentenceTransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def load_caches() -> (Dict[int, Any], Dict[int, str]):
    embed_cache = {}
    id_map = {}
    if EMBED_CACHE_FILE.exists():
        with open(EMBED_CACHE_FILE, "rb") as f:
            embed_cache = pickle.load(f)
    if ID_CACHE_FILE.exists():
        with open(ID_CACHE_FILE, "rb") as f:
            id_map = pickle.load(f)
    return embed_cache, id_map


def save_caches(embed_cache: Dict[int, Any], id_map: Dict[int, str]) -> None:
    with open(EMBED_CACHE_FILE, "wb") as f:
        pickle.dump(embed_cache, f)
    with open(ID_CACHE_FILE, "wb") as f:
        pickle.dump(id_map, f)


# ----------------- Indexing pipeline -----------------
def build_and_upsert_all(
    data_folder: str = DATA_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    model_name: str = MODEL_NAME,
    embed_batch: int = EMBED_BATCH,
    upsert_batch: int = UPSERT_BATCH,
) -> None:
    print("Loading PDFs from", data_folder)
    documents = load_all_pdfs(folder := data_folder)
    if not documents:
        print("No PDFs found. Exiting.")
        return

    print("Splitting into chunks...")
    chunks = split_documents_to_chunks(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Total chunks: {len(chunks)}")

    model = get_sentence_transformer(model_name=model_name)
    embed_cache, id_map = load_caches()

    vectors_to_upsert = []

    for start in tqdm(range(0, len(chunks), embed_batch), desc="Embedding batches"):
        batch_docs = chunks[start: start + embed_batch]
        texts = [d.page_content for d in batch_docs]

        to_encode = []
        to_encode_idx = []
        to_encode_keys = []

        # check cache
        for i, txt in enumerate(texts):
            key = hash(txt)
            if key in embed_cache:
                vec = embed_cache[key]
                meta = dict(batch_docs[i].metadata)
                meta.update({"snippet": meta.get(
                    "snippet", txt[:400]), "text": txt})
                cid = id_map.get(key, _generate_id())
                id_map.setdefault(key, cid)
                # ensure vector is JSON-serializable (list)
                vec_list = vec.tolist() if hasattr(vec, "tolist") else vec
                vectors_to_upsert.append((cid, vec_list, meta))
            else:
                to_encode.append(txt)
                to_encode_idx.append(i)
                to_encode_keys.append(key)

        # encode missing
        if to_encode:
            embeddings = model.encode(
                to_encode, batch_size=64, convert_to_numpy=True, show_progress_bar=False)
            for j, emb in enumerate(embeddings):
                key = to_encode_keys[j]
                embed_cache[key] = emb
                idx_local = to_encode_idx[j]
                txt = texts[idx_local]
                meta = dict(batch_docs[idx_local].metadata)
                meta.update({"snippet": meta.get(
                    "snippet", txt[:400]), "text": txt})
                cid = id_map.get(key, _generate_id())
                id_map.setdefault(key, cid)
                vectors_to_upsert.append((cid, emb.tolist(), meta))

        # bulk upsert when enough
        if len(vectors_to_upsert) >= upsert_batch:
            _upsert_batch(vectors_to_upsert)
            print(f"Upserted {len(vectors_to_upsert)} vectors")
            vectors_to_upsert = []

    # final flush
    if vectors_to_upsert:
        _upsert_batch(vectors_to_upsert)
        print(f"Final upserted {len(vectors_to_upsert)} vectors")

    save_caches(embed_cache, id_map)
    print("Indexing completed.")


def _upsert_batch(items: List[tuple]) -> None:
    if not items:
        return
    # Pinecone expects (id, values, metadata)
    _index.upsert(vectors=items)


# ----------------- Retrieval wrapper -----------------
class DocSearch:
    """
    Minimal retrieval wrapper that encodes queries with the same model and returns
    list of langchain.schema.Document objects (text in metadata.text or snippet).
    """

    def __init__(self, index=_index, model_name: str = MODEL_NAME):
        self._index = index
        self._model = get_sentence_transformer(model_name)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        q_vec = self._model.encode(query, convert_to_numpy=True).tolist()
        res = self._index.query(queries=[q_vec], top_k=k, include=[
                                "metadata", "values"])
        # adapt to different response shapes
        matches = getattr(res, "matches", None) or (res[0].matches if isinstance(
            res, list) and res else None) or res.get("matches", [])
        documents = []
        if not matches:
            return documents
        for m in matches:
            meta = getattr(m, "metadata", None) or m.get(
                "metadata", {}) if isinstance(m, dict) else {}
            text = meta.get("text") or meta.get("snippet") or ""
            documents.append(Document(page_content=text, metadata=meta))
        return documents


# ----------------- CLI -----------------
if __name__ == "__main__":
    # run indexing when executed directly
    build_and_upsert_all(data_folder=DATA_DIR)
    print("\nIndexing finished. Example usage:\n"
          "from fast_store_index import DocSearch\n"
          "ds = DocSearch()\n"
          "docs = ds.similarity_search('acne treatment', k=5)\n"
          "for d in docs: print(d.metadata.get('source'), d.metadata.get('snippet'))")
