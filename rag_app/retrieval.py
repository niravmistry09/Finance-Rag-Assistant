import os
from functools import cached_property

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from rag_app.settings import Settings, settings


class RAGRetriever:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg

    @cached_property
    def embeddings(self):
        return HuggingFaceEmbeddings(model_name=self.cfg.embedding_model_name)

    @cached_property
    def vector_store(self):
        return Chroma(
            persist_directory=self.cfg.persist_directory,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    @cached_property
    def reranker(self):
        return CrossEncoder(self.cfg.reranker_model_name)

    @cached_property
    def bm25_retriever(self):
        raw_data = self.vector_store.get()
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(raw_data["documents"], raw_data["metadatas"])
            if text
        ]
        if not documents:
            raise RuntimeError(
                "Vector store is empty. Run ingestion first: python main.py ingest"
            )

        retriever = BM25Retriever.from_documents(documents)
        retriever.k = self.cfg.top_k_bm25
        return retriever

    def retrieve(self, query: str):
        vector_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.cfg.top_k_vector,
                "fetch_k": self.cfg.mmr_fetch_k,
            },
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[self.cfg.bm25_weight, self.cfg.vector_weight],
        )

        retrieved_docs = ensemble_retriever.invoke(query)
        if not retrieved_docs:
            return []

        pairs = [(query, doc.page_content) for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)
        scored_docs = list(zip(scores, retrieved_docs))
        reranked_docs = sorted(scored_docs, key=lambda item: item[0], reverse=True)
        return [doc for _, doc in reranked_docs[: self.cfg.final_top_k]]


def citation_for(doc):
    source = os.path.basename(doc.metadata.get("source", "Unknown"))
    page = doc.metadata.get("page", "N/A")
    page_num = page + 1 if isinstance(page, int) else page
    return f"{source} (Page {page_num})"

