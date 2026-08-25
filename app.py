# app.py

import streamlit as st
import requests
import os

# ─── PAGE CONFIG ───
st.set_page_config(
    page_title="FinanceAI",
    page_icon="💼",
    layout="wide"
)

# ─── SIDEBAR ───
with st.sidebar:
    st.title("💼 FinanceAI")
    st.caption("Private offline AI assistant for financial services")

    st.divider()

    model = st.selectbox(
        "Select a model",
        options=["mistral", "llama3.2", "phi3:mini", "gemini"],
        help="First three models run locally. Gemini uses cloud API."
    )

    if model == "gemini":
        st.warning("☁️ Gemini — query sent to Google servers")
    else:
        st.success("🔒 Local model — no data leaves your device")

    st.divider()
    st.caption("Built with Ollama + FastAPI + Streamlit")

# ─── TABS ───
tab1, tab2, tab3 = st.tabs([
    "💬 Ask a Question",
    "📄 Ask My Document",
    "📊 Benchmark Results"
])

# ─── TAB 1: Live Chat ───
with tab1:
    st.subheader("Ask a finance question")
    st.caption("Select a model from the sidebar and type your question below")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                col1, col2, col3 = st.columns(3)
                col1.metric("Model", msg["meta"]["model"])
                col2.metric("Time", f"{msg['meta']['seconds']}s")
                col3.metric("RAM", f"{msg['meta']['ram_mb']} MB")

    question = st.chat_input("Type your finance question here...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner(f"Asking {model}..."):
                try:
                    result = requests.post(
                        "http://127.0.0.1:8000/ask",
                        json={"text": question, "model": model},
                        timeout=300
                    ).json()

                    st.write(result["answer"])

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Model", result["model"])
                    col2.metric("Time", f"{result['seconds']}s")
                    col3.metric("RAM", f"{result['ram_mb']} MB")

                    if not result.get("is_local"):
                        st.warning("⚠️ This question was sent to Gemini's servers")
                    else:
                        st.success("🔒 Processed entirely on your device")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "meta": {
                            "model": result["model"],
                            "seconds": result["seconds"],
                            "ram_mb": result["ram_mb"]
                        }
                    })

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

# ─── TAB 2: Document Chat ───
with tab2:
    st.subheader("Ask a question about your document")
    st.caption("Upload a financial PDF — it never leaves your device when using local models")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Client reports, compliance documents, contracts, policy documents"
    )

    if uploaded_file:
        st.success(f"✅ Document loaded: {uploaded_file.name}")

        if "doc_messages" not in st.session_state:
            st.session_state.doc_messages = []

        for msg in st.session_state.doc_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg["role"] == "assistant" and "meta" in msg:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Model", msg["meta"]["model"])
                    col2.metric("Time", f"{msg['meta']['seconds']}s")
                    col3.metric("RAM", f"{msg['meta']['ram_mb']} MB")

        doc_question = st.chat_input(
            "Ask a question about this document...",
            key="doc_input"
        )

        if doc_question:
            with st.chat_message("user"):
                st.write(doc_question)

            st.session_state.doc_messages.append({
                "role": "user",
                "content": doc_question
            })

            with st.chat_message("assistant"):
                with st.spinner(f"Reading document and asking {model}..."):
                    try:
                        result = requests.post(
                            "http://127.0.0.1:8000/ask-document",
                            files={"file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf"
                            )},
                            data={
                                "question": doc_question,
                                "model": model
                            },
                            timeout=300
                        ).json()

                        st.write(result["answer"])

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Model", result["model"])
                        col2.metric("Time", f"{result['seconds']}s")
                        col3.metric("RAM", f"{result['ram_mb']} MB")

                        if not result.get("is_local"):
                            st.warning("⚠️ This question was sent to Gemini's servers")
                        else:
                            st.success("🔒 Document processed entirely on your device")

                        st.session_state.doc_messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "meta": {
                                "model": result["model"],
                                "seconds": result["seconds"],
                                "ram_mb": result["ram_mb"]
                            }
                        })

                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
    else:
        st.info("Upload a PDF above to start asking questions about it.")
        st.markdown("""
        **Example use cases:**
        - Upload a client report and ask for a summary
        - Upload a compliance document and check specific rules
        - Upload a contract and ask about key terms
        - Upload internal policy documents and query them privately
        """)

# ─── TAB 3: Benchmark Results ───
with tab3:
    st.subheader("Benchmark Results")
    st.caption("Run benchmark.py first to see results here")

    if os.path.exists("results_scored.csv"):
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv("results_scored.csv")
        avg = df.groupby("model").mean(numeric_only=True).reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                avg, x="model", y="seconds",
                title="Average Response Time (lower is better)",
                color="model",
                labels={"seconds": "Seconds", "model": "Model"}
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                avg, x="model", y="quality_score",
                title="Average Quality Score (higher is better)",
                color="model",
                labels={"quality_score": "Score (1-5)", "model": "Model"}
            )
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(
            avg, x="seconds", y="quality_score",
            text="model",
            title="Speed vs Quality Tradeoff",
            labels={
                "seconds": "Response Time (s)",
                "quality_score": "Quality Score"
            }
        )
        fig3.update_traces(textposition="top center", marker_size=12)
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        st.subheader("Results by Category")
        if "category" in df.columns:
            cat_avg = df.groupby(["model", "category"])["quality_score"].mean().round(2).reset_index()
            fig4 = px.bar(
                cat_avg,
                x="category",
                y="quality_score",
                color="model",
                barmode="group",
                title="Quality Score by Category and Model",
                labels={
                    "quality_score": "Score (1-5)",
                    "category": "Category"
                }
            )
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("No benchmark results found yet. Run benchmark.py first to generate results.")