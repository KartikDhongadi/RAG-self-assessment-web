"""
Streamlit UI for the RAG pipeline with retrieval self-assessment.

The point of this UI is not just to show the answer — it's to show the
pipeline *deciding*: what it retrieved, which chunks it judged relevant,
when it rewrote the query, and when it refused to answer at all.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import os
import re
import uuid

import streamlit as st

import rag_core
from rag_core import LLMError

st.set_page_config(
    page_title="RAG with Retrieval Self-Assessment",
    page_icon="🔍",
    layout="wide",
)

# Streamlit secrets -> environment, so rag_core can read them uniformly.
# A missing secrets.toml is fine (e.g. local Ollama runs use env vars instead).
try:
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "LLM_PROVIDER"):
        if key in st.secrets and st.secrets[key]:
            os.environ[key] = str(st.secrets[key])
except FileNotFoundError:
    pass
except Exception:  # noqa: BLE001 - StreamlitSecretNotFoundError on newer versions
    pass

rag_core.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
rag_core.GROQ_MODEL = os.getenv("GROQ_MODEL", rag_core.GROQ_MODEL)


# ---------------------------------------------------------------------------
# Index (built once per container, then cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Embedding documents and building the vector index…")
def load_collection():
    return rag_core.setup_chroma()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline settings")

    top_k = st.slider("Chunks retrieved per iteration", 3, 10, 5)
    max_iterations = st.slider("Max retrieval iterations", 1, 5, 3)
    min_relevant = st.slider("Min relevant chunks to answer", 1, 5, 1)

    st.divider()
    st.caption(f"**Model:** `{rag_core.active_model_name()}`")
    st.caption(f"**Embeddings:** `all-MiniLM-L6-v2` (ONNX)")
    st.caption(f"**Vector store:** ChromaDB, cosine")

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Retrieve top-k chunks\n"
        "2. An LLM grades each chunk for genuine relevance\n"
        "3. Enough relevant context → answer, grounded and cited\n"
        "4. Not enough → rewrite the query and retrieve again\n"
        "5. Out of attempts → refuse rather than hallucinate"
    )

    st.divider()
    st.caption(
        "Public demo on a free tier. If you hit a rate-limit error, "
        "wait a moment and retry."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("RAG with Retrieval Self-Assessment")
st.markdown(
    "A RAG pipeline that **grades its own retrieved context before answering** — "
    "rewriting the query and re-retrieving when the context falls short, and "
    "refusing to answer rather than guessing."
)

try:
    demo_collection, demo_status = load_collection()
except FileNotFoundError as exc:
    st.error(f"Index could not be built: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface anything that breaks the cold start
    st.error(f"Unexpected error building the index: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Visitor uploads (session-only)
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]

with st.sidebar:
    st.divider()
    st.header("Your own documents")
    uploads = st.file_uploader(
        f"Upload up to {rag_core.MAX_UPLOAD_FILES} files "
        f"({rag_core.MAX_UPLOAD_MB} MB each). Kept only for this visit.",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploads:
        upload_key = tuple((f.name, f.size) for f in uploads)
        if st.session_state.get("upload_key") != upload_key:
            try:
                with st.spinner("Indexing your files…"):
                    st.session_state.upload_index = rag_core.build_upload_collection(
                        [(f.name, f.getvalue()) for f in uploads],
                        st.session_state.session_id,
                    )
                st.session_state.upload_key = upload_key
            except ValueError as exc:
                st.session_state.pop("upload_index", None)
                st.session_state.pop("upload_key", None)
                st.error(str(exc))
    else:
        st.session_state.pop("upload_index", None)
        st.session_state.pop("upload_key", None)

    has_uploads = "upload_index" in st.session_state
    source = st.radio(
        "Search in",
        ["Demo documents", "My uploads"],
        index=1 if has_uploads else 0,
        disabled=not has_uploads,
    )

if source == "My uploads" and "upload_index" in st.session_state:
    collection, status = st.session_state.upload_index
else:
    collection, status = demo_collection, demo_status

col_a, col_b, col_c = st.columns(3)
col_a.metric("Documents indexed", len(status.files))
col_b.metric("Chunks in vector store", status.total_chunks)
col_c.metric("Retrieval iterations allowed", max_iterations)

with st.expander(f"Corpus ({source.lower()}) — {len(status.files)} file(s)"):
    for name in status.files:
        st.markdown(f"- `{name}`")

st.divider()


# ---------------------------------------------------------------------------
# Trace rendering
# ---------------------------------------------------------------------------

# Models sometimes write LaTeX as \( ... \) or \[ ... \]; Streamlit only renders
# math inside $ ... $ and $$ ... $$, so convert before display.
_MATH_BLOCK = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def fix_math(text: str) -> str:
    text = _MATH_BLOCK.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return _MATH_INLINE.sub(lambda m: f"${m.group(1).strip()}$", text)


def friendly_error(exc: Exception) -> str:
    """Turn provider errors into a message that says what actually went wrong."""
    name = type(exc).__name__
    if name == "NotFoundError" or "model_not_found" in str(exc):
        return (
            f"The model `{rag_core.active_model_name()}` isn't available for this API key. "
            "Set `GROQ_MODEL` in your secrets to a model your key can use "
            "(see console.groq.com/docs/models)."
        )
    if name == "RateLimitError":
        return "Groq's free-tier rate limit was reached. Wait a few seconds and try again."
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "The Groq API key was rejected. Check `GROQ_API_KEY` in your secrets."
    if name in ("APIConnectionError", "APITimeoutError", "ConnectionError"):
        return "Couldn't reach the LLM provider. Check your connection and try again."
    return f"{name}: {exc}"


def render_chunk(chunk: dict, verdict: str | None = None) -> None:
    badge = ""
    if verdict == "relevant":
        badge = " :green-background[RELEVANT]"
    elif verdict == "rejected":
        badge = " :red-background[REJECTED]"
    st.markdown(f"**`{chunk['source']}` · page {chunk['page']}**{badge}")
    preview = chunk["text"][:400]
    st.caption(preview + ("…" if len(chunk["text"]) > 400 else ""))


def run_query(question: str) -> dict:
    """Run the pipeline, rendering the trace live. Returns the turn record."""
    grounding: list[dict] = []
    refused = False
    trace_log: list[str] = []

    with st.status("Running retrieval self-assessment…", expanded=True) as status_box:
        try:
            events = rag_core.run_pipeline(
                collection,
                question,
                top_k=top_k,
                max_iterations=max_iterations,
                min_relevant=min_relevant,
            )

            for event in events:
                if event.kind == "retrieve":
                    st.markdown(
                        f"**Iteration {event.iteration} — retrieving**  \n"
                        f"Query: `{event.query}`"
                    )
                    trace_log.append(f"Iteration {event.iteration}: retrieved for `{event.query}`")

                elif event.kind == "grade":
                    kept = len(event.relevant_indices)
                    total = len(event.chunks)
                    st.markdown(f"**Grading {total} chunk(s)** → {kept} judged relevant")
                    for i, chunk in enumerate(event.chunks):
                        verdict = "relevant" if i in event.relevant_indices else "rejected"
                        render_chunk(chunk, verdict)
                    trace_log.append(f"Graded {kept}/{total} chunks relevant")

                elif event.kind == "rewrite":
                    st.markdown(
                        f"Insufficient context — **rewriting query** to:  \n`{event.query}`"
                    )
                    trace_log.append(f"Rewrote query to: {event.query}")

                elif event.kind == "warning":
                    st.warning(event.message)

                elif event.kind == "answer":
                    grounding = event.chunks
                    status_box.update(
                        label=f"Grounded in {len(grounding)} chunk(s) — generating answer",
                        state="complete",
                    )

                elif event.kind == "refuse":
                    refused = True
                    status_box.update(
                        label="No sufficient context found — declining to answer",
                        state="error",
                    )

        except LLMError as exc:
            status_box.update(label="LLM not configured", state="error")
            st.error(str(exc))
            return {"question": question, "answer": None, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            status_box.update(label="Pipeline error", state="error")
            message = friendly_error(exc)
            st.error(message)
            return {"question": question, "answer": None, "error": message}

    if refused:
        st.warning(rag_core.REFUSAL)
        return {
            "question": question,
            "answer": rag_core.REFUSAL,
            "refused": True,
            "trace": trace_log,
            "sources": [],
        }

    try:
        placeholder = st.empty()
        buffer = ""
        for token in rag_core.answer_stream(question, grounding):
            buffer += token
            placeholder.markdown(fix_math(buffer) + "▌")
        answer = fix_math(buffer)
        placeholder.markdown(answer)
    except Exception as exc:  # noqa: BLE001
        message = f"Generation failed. {friendly_error(exc)}"
        st.error(message)
        return {"question": question, "answer": None, "error": message}

    sources = sorted({f"{c['source']} p.{c['page']}" for c in grounding})
    if sources:
        st.caption("Sources: " + " · ".join(f"`{s}`" for s in sources))

    return {
        "question": question,
        "answer": answer,
        "refused": False,
        "trace": trace_log,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

if "turns" not in st.session_state:
    st.session_state.turns = []

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
        else:
            if turn.get("refused"):
                st.warning(turn["answer"])
            else:
                st.markdown(turn["answer"])
            if turn.get("trace"):
                with st.expander("Retrieval trace"):
                    for line in turn["trace"]:
                        st.markdown(f"- {line}")
            if turn.get("sources"):
                st.caption("Sources: " + " · ".join(f"`{s}`" for s in turn["sources"]))

question = st.chat_input("Ask something about the indexed documents…")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        record = run_query(question)
    st.session_state.turns.append(record)
