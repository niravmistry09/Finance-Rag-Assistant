import argparse
import json
import subprocess
import sys


def run_ingest(args):
    from rag_app.ingestion import ingest

    result = ingest(force=args.force)
    print(json.dumps(result, indent=2))


def run_ask(args):
    from rag_app.generation import RAGService

    service = RAGService()
    try:
        result = service.answer(args.question)
        print("\n=== Answer ===\n")
        print(result.answer)
        print("\n=== Sources ===\n")
        for source in result.sources:
            print(f"- {source}")
    finally:
        service.close()


def run_eval(_args):
    from rag_app.evaluation import run_evaluation

    result = run_evaluation()
    print("\n=== Evaluation Results ===\n")
    print(result)


def run_api(args):
    import uvicorn

    uvicorn.run(
        "rag_app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def run_ui(args):
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    subprocess.run(command, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Finance RAG project runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Incrementally index new or changed PDFs",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the full vector database",
    )
    ingest_parser.set_defaults(func=run_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask one RAG question")
    ask_parser.add_argument("question", help="Question to answer from documents")
    ask_parser.set_defaults(func=run_ask)

    eval_parser = subparsers.add_parser("evaluate", help="Run RAGAS evaluation")
    eval_parser.set_defaults(func=run_eval)

    api_parser = subparsers.add_parser("api", help="Start FastAPI app")
    api_parser.add_argument("--host", default="0.0.0.0", help="API bind host")
    api_parser.add_argument("--port", type=int, default=8000, help="API bind port")
    api_parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart API automatically when source files change",
    )
    api_parser.set_defaults(func=run_api)

    ui_parser = subparsers.add_parser("ui", help="Start Streamlit web UI")
    ui_parser.add_argument("--host", default="127.0.0.1", help="UI bind host")
    ui_parser.add_argument("--port", type=int, default=8501, help="UI bind port")
    ui_parser.set_defaults(func=run_ui)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
