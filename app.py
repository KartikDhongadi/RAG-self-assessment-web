"""
Streamlit UI for the RAG pipeline with retrieval self-assessment.

Designed for people with no AI background: plain language up front, sample
questions to get started, and the technical detail (how each answer was
found) available but collapsed.

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
    page_title="Ask the Documents",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
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
# Content
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    "What is retrieval-augmented generation (RAG)?",
    "How does the Transformer model use attention?",
    "How can deep learning detect network intrusions?",
    "What is fog computing?",
    "How is Frankenstein's creature described?",
]
CANT_ANSWER_QUESTION = "Who won the 2022 FIFA World Cup?"

HOW_IT_WORKS = """
1. **It searches.** When you ask a question, the app looks through the documents
   for passages that might contain the answer.
2. **It checks.** An AI reads those passages and decides which ones actually help.
   If none do, it rephrases your question and searches again.
3. **It answers honestly.** It answers using only the helpful passages and shows
   where each fact came from. If it still can't find anything, it tells you,
   instead of making something up.
"""

FRIENDLY_REFUSAL = (
    "I couldn't find the answer to that in these documents, so I won't guess. "
    "Try rephrasing your question, or ask about one of the topics in the document list."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Getting the documents ready (this takes a minute the first time)…")
def load_collection():
    return rag_core.setup_chroma()


# Models sometimes write LaTeX as \( ... \) or \[ ... \]; Streamlit only renders
# math inside $ ... $ and $$ ... $$, so convert before display.
_MATH_BLOCK = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def fix_math(text: str) -> str:
    text = _MATH_BLOCK.sub(lambda m: f"$${m.group(1).strip()}$$", text)
    return _MATH_INLINE.sub(lambda m: f"${m.group(1).strip()}$", text)


def friendly_error(exc: Exception) -> str:
    """Plain-language error for visitors, with owner details where useful."""
    name = type(exc).__name__
    if name == "RateLimitError":
        return (
            "The app is busy right now (it runs on a free usage allowance). "
            "Please wait a minute and try again."
        )
    if name in ("APIConnectionError", "APITimeoutError", "ConnectionError"):
        return "The app couldn't reach its AI service. Please try again in a moment."
    if name == "NotFoundError" or "model_not_found" in str(exc):
        detail = (
            f"model `{rag_core.active_model_name()}` isn't available for this API key; "
            "set `GROQ_MODEL` in the secrets"
        )
    elif name in ("AuthenticationError", "PermissionDeniedError"):
        detail = "the Groq API key was rejected; check `GROQ_API_KEY` in the secrets"
    else:
        detail = f"{name}: {exc}"
    return f"Something went wrong on our side. Please try again later.\n\n*For the app owner: {detail}.*"


def sources_line(sources: list[str]) -> str:
    return "📄 **Sources:** " + " · ".join(sources)


def ask(question: str) -> None:
    """Queue a question (used by the sample-question buttons)."""
    st.session_state.pending_question = question


def start_over() -> None:
    st.session_state.turns = []


# ---------------------------------------------------------------------------
# Advanced settings (sidebar, collapsed by default)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Advanced settings")
    st.caption("You don't need to change these. They control how hard the app searches.")
    top_k = st.slider(
        "Passages checked per search", 3, 10, 5,
        help="How many passages the app reads each time it searches.",
    )
    max_iterations = st.slider(
        "Searches before giving up", 1, 5, 3,
        help="If the first search finds nothing useful, the app rephrases the question "
             "and tries again, up to this many times.",
    )
    min_relevant = st.slider(
        "Useful passages needed to answer", 1, 5, 1,
        help="The app only answers once it has found at least this many useful passages.",
    )
    st.divider()
    st.caption(
        f"Model: `{rag_core.active_model_name()}` · Embeddings: all-MiniLM-L6-v2 · "
        "Vector store: ChromaDB"
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📚 Ask the Documents")
st.markdown("##### Your documents, answered honestly.")
st.markdown(
    "Ask anything in plain English and get an answer drawn straight from a library of "
    "research papers and classic books, **with the exact page it came from**. If the answer "
    "isn't there, it tells you instead of making something up. You can even upload your own files."
)

try:
    demo_collection, demo_status = load_collection()
except FileNotFoundError as exc:
    st.error(f"The documents couldn't be loaded: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 - surface anything that breaks the cold start
    st.error(f"The documents couldn't be loaded: {exc}")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "turns" not in st.session_state:
    st.session_state.turns = []


# ---------------------------------------------------------------------------
# Documents: sample set or the visitor's own files
# ---------------------------------------------------------------------------

with st.expander("📎 Ask about your own files instead"):
    uploads = st.file_uploader(
        "Add PDF, Word, text, CSV or Excel files",
        type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help=f"Up to {rag_core.MAX_UPLOAD_FILES} files, {rag_core.MAX_UPLOAD_MB} MB each.",
    )
    st.caption(
        f"Up to {rag_core.MAX_UPLOAD_FILES} files, {rag_core.MAX_UPLOAD_MB} MB each. "
        "Your files are only used while this page is open and are never saved."
    )

    if uploads:
        upload_key = tuple((f.name, f.size) for f in uploads)
        if st.session_state.get("upload_key") != upload_key:
            try:
                with st.spinner("Reading your files…"):
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
source = "Sample documents"
if has_uploads:
    source = st.radio(
        "Answer questions from:",
        ["My files", "Sample documents"],
        horizontal=True,
    )

if source == "My files" and has_uploads:
    collection, status = st.session_state.upload_index
else:
    collection, status = demo_collection, demo_status

titles = sorted(rag_core.display_name(name) for name in status.files)
noun = "file" if source == "My files" else "sample document"
label = f"{len(titles)} {noun}{'' if len(titles) == 1 else 's'}"
if source == "My files":
    label = label.replace("file", "of your files", 1) if len(titles) > 1 else "your file"
with st.expander(f"📄 Searching {label} (see list)"):
    for title in titles:
        st.markdown(f"- {title}")

with st.expander("💡 How does this work?"):
    st.markdown(HOW_IT_WORKS)


# ---------------------------------------------------------------------------
# Answering a question
# ---------------------------------------------------------------------------

def render_passage(chunk: dict, useful: bool) -> None:
    verdict = ":green[✔ Useful]" if useful else ":gray[✖ Not relevant]"
    st.markdown(f"{verdict} — *{chunk['title']}*, page {chunk['page']}")
    preview = chunk["text"][:300]
    st.caption(preview + ("…" if len(chunk["text"]) > 300 else ""))


def run_query(question: str) -> dict:
    """Run the pipeline with a plain-language progress trail. Returns the turn record."""
    grounding: list[dict] = []
    refused = False
    trace: list[str] = []

    with st.status("Looking through the documents…", expanded=False) as status_box:
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
                    line = (
                        f"🔎 Searched the documents for: *{event.query}*"
                        if event.iteration == 1
                        else f"🔎 Searched again for: *{event.query}*"
                    )
                    st.markdown(line)
                    trace.append(line)

                elif event.kind == "grade":
                    kept, total = len(event.relevant_indices), len(event.chunks)
                    line = f"📖 Checked {total} passages: **{kept} useful**"
                    st.markdown(line)
                    trace.append(line)
                    for i, chunk in enumerate(event.chunks):
                        render_passage(chunk, i in event.relevant_indices)

                elif event.kind == "rewrite":
                    line = "✏️ Not enough useful information yet, so the question was rephrased."
                    st.markdown(line)
                    trace.append(line)

                elif event.kind == "warning":
                    st.caption(event.message)

                elif event.kind == "answer":
                    grounding = event.chunks
                    status_box.update(
                        label="See how this answer was found",
                        state="complete",
                        expanded=False,
                    )

                elif event.kind == "refuse":
                    refused = True
                    status_box.update(
                        label="See what was searched",
                        state="complete",
                        expanded=False,
                    )

        except LLMError as exc:
            status_box.update(label="Something went wrong", state="error", expanded=False)
            message = f"The app isn't set up correctly right now.\n\n*For the app owner: {exc}*"
            st.error(message)
            return {"question": question, "error": message}
        except Exception as exc:  # noqa: BLE001
            status_box.update(label="Something went wrong", state="error", expanded=False)
            message = friendly_error(exc)
            st.error(message)
            return {"question": question, "error": message}

    if refused:
        st.info(FRIENDLY_REFUSAL, icon="🤷")
        return {"question": question, "answer": FRIENDLY_REFUSAL, "refused": True,
                "trace": trace, "sources": []}

    try:
        placeholder = st.empty()
        buffer = ""
        for token in rag_core.answer_stream(question, grounding):
            buffer += token
            placeholder.markdown(fix_math(buffer) + "▌")
        answer = fix_math(buffer)
        placeholder.markdown(answer)
    except Exception as exc:  # noqa: BLE001
        message = friendly_error(exc)
        st.error(message)
        return {"question": question, "error": message}

    sources = []
    for c in grounding:
        label = f"{c['title']} (p. {c['page']})"
        if label not in sources:
            sources.append(label)
    if sources:
        st.caption(sources_line(sources))

    return {"question": question, "answer": answer, "refused": False,
            "trace": trace, "sources": sources}


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

pending = st.session_state.pop("pending_question", None)

if not st.session_state.turns and not pending:
    st.markdown("#### Not sure what to ask? Try one of these:")
    cols = st.columns(2)
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        cols[i % 2].button(sample, on_click=ask, args=(sample,), use_container_width=True)
    cols[len(SAMPLE_QUESTIONS) % 2].button(
        f"🚫 {CANT_ANSWER_QUESTION}",
        on_click=ask, args=(CANT_ANSWER_QUESTION,), use_container_width=True,
        help="This isn't in any of the documents. Try it to see the app refuse to guess.",
    )
    st.caption("🚫 The last one isn't covered by the documents. Try it to see the app say so instead of guessing.")

for turn in st.session_state.turns:
    with st.chat_message("user", avatar="🙂"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="📚"):
        if turn.get("error"):
            st.error(turn["error"])
            continue
        if turn.get("trace"):
            with st.expander("See how this answer was found"):
                for line in turn["trace"]:
                    st.markdown(line)
        if turn.get("refused"):
            st.info(turn["answer"], icon="🤷")
        else:
            st.markdown(turn["answer"])
        if turn.get("sources"):
            st.caption(sources_line(turn["sources"]))

typed = st.chat_input("Type your question here…")
question = typed or pending

if question:
    with st.chat_message("user", avatar="🙂"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📚"):
        record = run_query(question)
    st.session_state.turns.append(record)

if st.session_state.turns:
    st.button("↺ Start over", on_click=start_over)
