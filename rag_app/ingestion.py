import hashlib
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.settings import Settings, settings


MANIFEST_FILENAME = "ingestion_manifest.json"


def get_embeddings(cfg: Settings):
    return HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)


def get_vector_store(cfg: Settings = settings):
    return Chroma(
        persist_directory=cfg.persist_directory,
        embedding_function=get_embeddings(cfg),
        collection_metadata={"hnsw:space": "cosine"},
    )


def manifest_path(cfg: Settings):
    return Path(cfg.persist_directory).parent / MANIFEST_FILENAME


def load_manifest(cfg: Settings):
    path = manifest_path(cfg)
    if not path.exists():
        return {"version": 1, "documents": {}}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_manifest(manifest, cfg: Settings):
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_pdf_files(docs_path: str):
    path = Path(docs_path)
    if not path.exists():
        raise FileNotFoundError(f"Docs directory does not exist: {path}")

    pdf_files = sorted(path.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {path}")

    return pdf_files


def load_pdf(path: Path):
    loader = PyPDFLoader(str(path))
    return loader.load()


def split_documents(documents, cfg: Settings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    return splitter.split_documents(documents)


def enrich_chunks(chunks, pdf_path: Path, doc_id: str, file_hash: str):
    enriched_chunks = []
    chunk_ids = []

    for index, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}:{file_hash[:16]}:{index}"
        chunk.metadata = {
            **(chunk.metadata or {}),
            "source": str(pdf_path),
            "source_filename": pdf_path.name,
            "source_path": str(pdf_path.resolve()),
            "source_doc_id": doc_id,
            "source_file_hash": file_hash,
            "chunk_index": index,
        }
        enriched_chunks.append(chunk)
        chunk_ids.append(chunk_id)

    return enriched_chunks, chunk_ids


def seed_manifest_for_existing_index(manifest, cfg: Settings, vector_store):
    """Mark current PDFs as already indexed when adopting an older Chroma DB."""
    if manifest["documents"]:
        return manifest

    if vector_store._collection.count() == 0:
        return manifest

    for pdf_path in list_pdf_files(cfg.docs_path):
        doc_id = str(pdf_path.resolve())
        manifest["documents"][doc_id] = {
            "filename": pdf_path.name,
            "path": doc_id,
            "file_hash": file_sha256(pdf_path),
            "chunk_ids": [],
            "legacy_seeded": True,
        }

    save_manifest(manifest, cfg)
    return manifest


def delete_document_chunks(vector_store, chunk_ids):
    if chunk_ids:
        vector_store.delete(ids=chunk_ids)


def index_pdf(vector_store, pdf_path: Path, cfg: Settings):
    doc_id = str(pdf_path.resolve())
    file_hash = file_sha256(pdf_path)
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages, cfg)
    enriched_chunks, chunk_ids = enrich_chunks(chunks, pdf_path, doc_id, file_hash)

    if enriched_chunks:
        vector_store.add_documents(enriched_chunks, ids=chunk_ids)

    return {
        "filename": pdf_path.name,
        "path": doc_id,
        "file_hash": file_hash,
        "chunk_ids": chunk_ids,
        "legacy_seeded": False,
    }


def ingest(force: bool = False, cfg: Settings = settings):
    persist_path = Path(cfg.persist_directory)
    manifest = {"version": 1, "documents": {}}

    if force and persist_path.exists():
        shutil.rmtree(persist_path)

    if force and manifest_path(cfg).exists():
        manifest_path(cfg).unlink()

    vector_store = get_vector_store(cfg)
    manifest = load_manifest(cfg)
    manifest = seed_manifest_for_existing_index(manifest, cfg, vector_store)

    added_files = []
    updated_files = []
    skipped_files = []
    warnings = []

    for pdf_path in list_pdf_files(cfg.docs_path):
        doc_id = str(pdf_path.resolve())
        file_hash = file_sha256(pdf_path)
        previous = manifest["documents"].get(doc_id)

        if previous and previous.get("file_hash") == file_hash:
            skipped_files.append(pdf_path.name)
            continue

        if previous:
            old_chunk_ids = previous.get("chunk_ids", [])
            if old_chunk_ids:
                delete_document_chunks(vector_store, old_chunk_ids)
            else:
                warnings.append(
                    f"{pdf_path.name} was tracked from a legacy index. "
                    "Run 'python main.py ingest --force' once if this file was edited."
                )
            updated_files.append(pdf_path.name)
        else:
            added_files.append(pdf_path.name)

        manifest["documents"][doc_id] = index_pdf(vector_store, pdf_path, cfg)

    save_manifest(manifest, cfg)

    return {
        "persist_directory": cfg.persist_directory,
        "document_chunks": vector_store._collection.count(),
        "added_files": added_files,
        "updated_files": updated_files,
        "skipped_files": skipped_files,
        "warnings": warnings,
        "forced_rebuild": force,
    }
