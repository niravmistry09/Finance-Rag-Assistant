from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

from rag_app.generation import RAGService
from rag_app.ingestion import ingest
from rag_app.settings import settings


st.set_page_config(
    page_title="Finance RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading retrieval and generation models...")
def get_service():
    return RAGService()


def reset_service_cache():
    service = st.session_state.get("_service")
    if service is not None:
        service.close()
    st.cache_resource.clear()
    st.session_state.pop("_service", None)


def get_cached_service():
    service = get_service()
    st.session_state["_service"] = service
    return service


if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_contexts" not in st.session_state:
    st.session_state.show_contexts = False


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    .source-pill {
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 0.35rem 0.5rem;
        margin: 0.15rem 0;
        background: #f6f8fa;
        color: #0A0A0A;
        font-size: 0.88rem;
    }
    .metric-box {
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.65rem;
        color: #0A0A0A;
        background: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.title("Finance RAG")

    st.caption("Current index")
    st.write(settings.persist_directory)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        st.toggle("Contexts", key="show_contexts")

    st.divider()

    st.caption("Documents")
    uploaded_files = st.file_uploader(
        "Add PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        docs_path = Path(settings.docs_path)
        docs_path.mkdir(parents=True, exist_ok=True)
        for uploaded_file in uploaded_files:
            target = docs_path / uploaded_file.name
            target.write_bytes(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s).")

    if st.button("Index New / Changed PDFs", type="primary", use_container_width=True):
        with st.spinner("Indexing only new or changed PDFs..."):
            result = ingest(force=False)
            reset_service_cache()
        st.success(
            "Index updated: "
            f"{len(result['added_files'])} added, "
            f"{len(result['updated_files'])} updated, "
            f"{len(result['skipped_files'])} skipped."
        )
        for warning in result.get("warnings", []):
            st.warning(warning)

    if st.button("Full Rebuild", use_container_width=True):
        with st.spinner("Rebuilding the full document index..."):
            result = ingest(force=True)
            reset_service_cache()
        st.success(f"Full rebuild complete. Indexed {result['document_chunks']} chunks.")

    st.divider()

    st.caption("Retrieval")
    st.markdown(
        f"""
        <div class="metric-box">
        Vector top-k: {settings.top_k_vector}<br>
        BM25 top-k: {settings.top_k_bm25}<br>
        Final contexts: {settings.final_top_k}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("Finance RAG Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            st.caption("Sources")
            for source in message["sources"]:
                st.markdown(
                    f'<div class="source-pill">{source}</div>',
                    unsafe_allow_html=True,
                )

        if (
            message["role"] == "assistant"
            and st.session_state.show_contexts
            and message.get("contexts")
        ):
            with st.expander("Retrieved contexts"):
                for index, context in enumerate(message["contexts"], start=1):
                    st.markdown(f"**Context {index}**")
                    st.write(context)


question = st.chat_input("Ask a question from the indexed PDFs")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Retrieving and generating answer..."):
            try:
                result = get_cached_service().answer(question)
                placeholder.markdown(result.answer)

                if result.sources:
                    st.caption("Sources")
                    for source in result.sources:
                        st.markdown(
                            f'<div class="source-pill">{source}</div>',
                            unsafe_allow_html=True,
                        )

                if st.session_state.show_contexts and result.contexts:
                    with st.expander("Retrieved contexts"):
                        for index, context in enumerate(result.contexts, start=1):
                            st.markdown(f"**Context {index}**")
                            st.write(context)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                        "contexts": result.contexts,
                    }
                )
            except Exception as exc:
                error_message = f"Request failed: {exc}"
                placeholder.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "contexts": [],
                    }
                )
