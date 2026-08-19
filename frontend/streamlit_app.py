"""
Streamlit Chat Frontend
========================
Persistent chat UI backed by Postgres memory via /chat endpoint.
"""

import streamlit as st
import requests  # type: ignore

API_URL = "http://127.0.0.1:8080"

st.set_page_config(page_title="GitHub Codebase RAG", page_icon="🔍", layout="wide")
st.title("🔍 GitHub Codebase RAG")

# ── Session state ────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []   # [{"role": "user"/"assistant", "content": str, "sources": []}]


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📦 Index a repository")
    repo_url = st.text_input("GitHub URL", placeholder="https://github.com/owner/repo")

    if st.button("Ingest", type="primary", use_container_width=True):
        if repo_url.strip():
            with st.spinner("Cloning, scanning, chunking, embedding..."):
                try:
                    resp = requests.post(f"{API_URL}/ingest", json={"url": repo_url}, timeout=600)
                    if resp.ok:
                        st.success("Repository indexed!")
                    else:
                        st.error(resp.json().get("detail", resp.text))
                except requests.exceptions.Timeout:
                    st.error("The ingestion request timed out. Processing the repository took longer than 10 minutes.")
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the API. Is the backend server running?")

    st.divider()

    if st.button("🆕 New conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Past sessions")
    try:
        sessions = requests.get(f"{API_URL}/sessions", timeout=5).json()
        for s in sessions[:10]:
            label = f"{s['repo_url'] or 'Untitled'} — {s['created_at'][:16]}"
            if st.button(label, key=s["id"], use_container_width=True):
                st.session_state.session_id = s["id"]
                history = requests.get(f"{API_URL}/sessions/{s['id']}/history", timeout=5).json()
                st.session_state.messages = history
                st.rerun()
    except requests.exceptions.ConnectionError:
        st.caption("API not reachable")


# ── Chat history display ────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Source files"):
                for src in msg["sources"]:
                    st.code(src, language="text")


# ── Chat input ───────────────────────────────────────────────
question = st.chat_input("Ask a question about the codebase...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "session_id": st.session_state.session_id},
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    st.session_state.session_id = data["session_id"]
                    st.write(data["answer"])

                    if data["sources"]:
                        with st.expander("Source files"):
                            for src in data["sources"]:
                                st.code(src, language="text")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                    })
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the API. Is `python main.py` running?")