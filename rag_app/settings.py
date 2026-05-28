import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    docs_path: str = os.getenv("DOCS_PATH", "docs")
    persist_directory: str = os.getenv("PERSIST_DIRECTORY", "db/chroma_db")

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"
    )
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

    top_k_vector: int = int(os.getenv("TOP_K_VECTOR", "8"))
    top_k_bm25: int = int(os.getenv("TOP_K_BM25", "8"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "3"))
    mmr_fetch_k: int = int(os.getenv("MMR_FETCH_K", "20"))

    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.4"))

    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    evaluation_dataset_path: str = os.getenv(
        "EVALUATION_DATASET_PATH", "evaluation_dataset.json"
    )
    evaluation_generated_rows_path: str = os.getenv(
        "EVALUATION_GENERATED_ROWS_PATH", "evaluation_generated_rows.json"
    )
    evaluation_results_path: str = os.getenv(
        "EVALUATION_RESULTS_PATH", "evaluation_results_scores.json"
    )

    ragas_timeout: int = int(os.getenv("RAGAS_TIMEOUT", "240"))
    ragas_max_retries: int = int(os.getenv("RAGAS_MAX_RETRIES", "8"))
    ragas_max_workers: int = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
    api_cooldown_seconds: float = float(os.getenv("API_COOLDOWN_SECONDS", "4"))


settings = Settings()

