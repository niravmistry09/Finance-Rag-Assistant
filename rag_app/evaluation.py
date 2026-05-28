import atexit
import json
import os
import time
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.run_config import RunConfig

from rag_app.generation import RAGService
from rag_app.runtime import close_google_model, get_google_api_keys
from rag_app.settings import Settings, settings


def build_evaluator_llm(cfg: Settings = settings):
    last_error = None
    for index, api_key in enumerate(get_google_api_keys(), start=1):
        try:
            evaluator_llm = ChatGoogleGenerativeAI(
                model=cfg.gemini_model_name,
                temperature=0,
                google_api_key=api_key,
            )
            evaluator_llm.invoke("ping")
            os.environ["GOOGLE_API_KEY"] = api_key
            atexit.register(close_google_model, evaluator_llm)
            return evaluator_llm
        except Exception as exc:
            last_error = exc
            print(f"[WARNING] Google API key #{index} failed: {exc}")

    raise RuntimeError(
        "Could not initialize evaluator LLM. Set GOOGLE_API_KEY, "
        "GOOGLE_API_KEY_1, or GOOGLE_API_KEY_2 in .env."
    ) from last_error


def load_evaluation_dataset(path: str):
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing evaluation dataset: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list) or not rows:
        raise ValueError("Evaluation dataset must contain a non-empty JSON list.")

    for index, row in enumerate(rows, start=1):
        if not row.get("question"):
            raise ValueError(f"Evaluation row {index} is missing 'question'.")
        if not (row.get("reference") or row.get("ground_truth") or row.get("answer")):
            raise ValueError(
                f"Evaluation row {index} needs one of: reference, ground_truth, answer."
            )

    return rows


def collect_rag_outputs(eval_rows, service: RAGService, cfg: Settings = settings):
    generated_rows = []

    for index, item in enumerate(eval_rows, start=1):
        question = item["question"]
        reference = item.get("reference") or item.get("ground_truth") or item["answer"]
        reference_contexts = item.get("reference_contexts")

        print(f"[{index}/{len(eval_rows)}] Question: {question}")
        result = service.answer(question)

        if not result.contexts:
            raise ValueError(
                "RAG returned no retrieved contexts for evaluation. "
                f"Question: {question}"
            )

        row = {
            "user_input": question,
            "response": result.answer,
            "retrieved_contexts": result.contexts,
            "reference": reference,
            "sources": result.sources,
        }
        if reference_contexts:
            row["reference_contexts"] = reference_contexts

        generated_rows.append(row)
        print(f"Generated answer:\n{result.answer}\n")
        print(f"Retrieved contexts: {len(result.contexts)}")
        print("-" * 80)
        time.sleep(cfg.api_cooldown_seconds)

    with Path(cfg.evaluation_generated_rows_path).open("w", encoding="utf-8") as file:
        json.dump(generated_rows, file, indent=4, ensure_ascii=False)

    return generated_rows


def run_evaluation(cfg: Settings = settings):
    evaluator_llm = None
    service = RAGService(cfg)
    try:
        evaluator_llm = build_evaluator_llm(cfg)
        evaluator_embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)
        eval_rows = load_evaluation_dataset(cfg.evaluation_dataset_path)
        generated_rows = collect_rag_outputs(eval_rows, service, cfg)

        time.sleep(10)
        result = evaluate(
            dataset=Dataset.from_list(generated_rows),
            metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=RunConfig(
                timeout=cfg.ragas_timeout,
                max_retries=cfg.ragas_max_retries,
                max_workers=cfg.ragas_max_workers,
            ),
        )

        try:
            scores = result.to_pandas().to_dict(orient="records")
            with Path(cfg.evaluation_results_path).open("w", encoding="utf-8") as file:
                json.dump(scores, file, indent=4, ensure_ascii=False)
        except Exception as exc:
            print(f"[WARNING] Could not save structured scores: {exc}")
            with Path(cfg.evaluation_results_path).open("w", encoding="utf-8") as file:
                file.write(str(result))

        return result
    finally:
        if evaluator_llm is not None:
            close_google_model(evaluator_llm)
        service.close()

