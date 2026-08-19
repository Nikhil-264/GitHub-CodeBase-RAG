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
if "active_repo" not in st.session_state:
    st.session_state.active_repo = None
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
                        data = resp.json()
                        st.success(f"Indexed repository '{data.get('repo')}'!")
                        if data.get("session_id"):
                            st.session_state.session_id = data["session_id"]
                            st.session_state.active_repo = data.get("repo")
                            st.session_state.messages = []
                            st.rerun()
                    else:
                        st.error(resp.json().get("detail", resp.text))
                except requests.exceptions.Timeout:
                    st.error("The ingestion request timed out. Processing the repository took longer than 10 minutes.")
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the API. Is the backend server running?")

    st.divider()

    if st.button("🆕 New conversation", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.active_repo = None
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ Clear All Data (Reset)", use_container_width=True):
        try:
            resp = requests.post(f"{API_URL}/reset", timeout=30)
            if resp.ok:
                st.session_state.session_id = None
                st.session_state.active_repo = None
                st.session_state.messages = []
                st.success("All chats, embeddings, and repos reset!")
                st.rerun()
            else:
                st.error("Reset failed on server.")
        except requests.exceptions.RequestException:
            st.error("Could not reach API to reset.")

    st.divider()
    st.caption("Past sessions")
    try:
        sessions = requests.get(f"{API_URL}/sessions", timeout=5).json()
        for s in sessions[:10]:
            repo_label = s.get('repo_name') or s.get('repo_url') or 'Global Scope'
            label = f"{repo_label} — {s['created_at'][:16]}"
            if st.button(label, key=s["id"], use_container_width=True):
                st.session_state.session_id = s["id"]
                st.session_state.active_repo = s.get("repo_name")
                history = requests.get(f"{API_URL}/sessions/{s['id']}/history", timeout=5).json()
                st.session_state.messages = history
                st.rerun()
    except requests.exceptions.RequestException:
        st.caption("⚠️ Backend API not reachable (start server via `python main.py`)")


# ── Active Scope Banner ──────────────────────────────────────
if st.session_state.active_repo:
    st.info(f"📌 **Active Repository Scope**: `{st.session_state.active_repo}` — Search queries are strictly scoped to this repository.")
else:
    st.caption("ℹ️ *No repository active for this session. Ingest or select a past session to scope questions to a repository.*")


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
                payload = {
                    "question": question,
                    "session_id": st.session_state.session_id,
                }
                if st.session_state.active_repo:
                    payload["repo_name"] = st.session_state.active_repo

                resp = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
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
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the API ({e}). Is `python main.py` running?")