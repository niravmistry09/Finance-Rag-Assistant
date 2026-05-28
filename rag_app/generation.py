import atexit
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_app.retrieval import RAGRetriever, citation_for
from rag_app.runtime import close_google_model, get_google_api_key
from rag_app.schemas import RAGResponse
from rag_app.settings import Settings, settings


class RAGService:
    def __init__(self, cfg: Settings = settings, retriever: RAGRetriever | None = None):
        self.cfg = cfg
        self.retriever = retriever or RAGRetriever(cfg)
        self._models = {}
        atexit.register(self.close)

    def _get_model(self):
        api_key = get_google_api_key()
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing. Set GOOGLE_API_KEY or GOOGLE_API_KEY_1 in .env."
            )

        model = self._models.get(api_key)
        if model is None:
            model = ChatGoogleGenerativeAI(
                model=self.cfg.gemini_model_name,
                temperature=self.cfg.temperature,
                google_api_key=api_key,
            )
            self._models[api_key] = model
        return model

    def close(self):
        for model in list(self._models.values()):
            close_google_model(model)
        self._models.clear()

    def answer(self, query: str) -> RAGResponse:
        final_docs = self.retriever.retrieve(query)
        if not final_docs:
            return RAGResponse(
                answer=(
                    "I don't have enough information to answer that question based "
                    "on the provided documents."
                ),
                sources=[],
                contexts=[],
                metadata=[],
            )

        context = self._format_context(final_docs)
        prompt = self._build_prompt(query=query, context=context)

        try:
            response = self._get_model().invoke(prompt)
            answer = response.content
        except Exception as exc:
            answer = f"Error during generation: {exc}"

        sources = []
        metadata = []
        for doc in final_docs:
            citation = citation_for(doc)
            if citation not in sources:
                sources.append(citation)
            metadata.append(doc.metadata)

        return RAGResponse(
            answer=answer,
            sources=sources,
            contexts=[doc.page_content for doc in final_docs],
            metadata=metadata,
        )

    @staticmethod
    def _format_context(docs):
        formatted_context = []
        for doc in docs:
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "N/A")
            page_num = page + 1 if isinstance(page, int) else page
            formatted_context.append(
                f"[Source: {source} | Page: {page_num}]\n{doc.page_content[:1500]}"
            )
        return "\n\n".join(formatted_context)

    @staticmethod
    def _build_prompt(query: str, context: str):
        return f"""
You are an expert financial and business research assistant.

Answer the user's question ONLY using the provided context.

Rules:
1. Be precise and professional.
2. Cite the source document and page when possible.
3. Do NOT hallucinate.
4. If information is missing, clearly say so.

Context:
{context}

Question:
{query}

Provide a clear and concise answer.
"""

