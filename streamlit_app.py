import streamlit as st
import time

from main import AgenticRAG

st.set_page_config(
    page_title="Know Your Laws - Agentic AI",
    page_icon="⚖️",
    layout="wide"
)

LEGAL_DOCS = [
    "data/legal_pdfs/Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf",
    "data/legal_pdfs/it_act_2000_updated.pdf",
    "data/legal_pdfs/CP Act 2019_1732700731.pdf"
]

# ---------------------------------------------------------------------- #
# Mode selection screen (shown until the user picks a mode)
# ---------------------------------------------------------------------- #

if "mode" not in st.session_state:

    st.title("⚖️ Know Your Laws - Agentic AI")

    st.subheader("How would you like your answers?")

    st.markdown(
        """
This assistant answers questions on Indian law using a legal corpus
(BNSS 2023, IT Act 2000, Consumer Protection Act 2019), and falls back
to web search when the corpus does not cover your question.
"""
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 👔 Professional Mode")

        st.markdown(
            """
For **lawyers and law students**.

- Detailed, fully bulleted answers
- Specific Act, provision and page cited for each point
- Precise legal terminology
"""
        )

        if st.button("Enter Professional Mode", use_container_width=True):

            st.session_state.mode = "professional"

            st.rerun()

    with col2:

        st.markdown("### 🧑 Normal Mode")

        st.markdown(
            """
For **anyone needing quick legal guidance**.

- Short, plain-language bullet-point overviews
- No legal jargon
- Sources named simply so you know where the info comes from
"""
        )

        if st.button("Enter Normal Mode", use_container_width=True):

            st.session_state.mode = "normal"

            st.rerun()

    st.stop()

# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #

with st.sidebar:

    st.title("⚖️ Know Your Laws")

    st.info(f"Current Mode: {st.session_state.mode.title()}")

    if st.button("🔄 Switch Mode"):

        del st.session_state["mode"]

        st.rerun()

    st.markdown("---")

    st.markdown(
        """
### About

Agentic Legal RAG System for Indian Laws.

Current Legal Corpus:

- Bharatiya Nagarik Suraksha Sanhita, 2023
- Information Technology Act, 2000
- Consumer Protection Act, 2019

Features:

- FAISS Vector Search
- Corrective Agentic RAG
- Context Evaluation Agent
- Tavily Search Fallback
- Conversational Query Rewriting
- Source Attribution
- Confidence Scoring
"""
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat History"):

        st.session_state.chat_history = []

        st.rerun()

st.title("⚖️ Know Your Laws - Agentic AI")

st.markdown(
    """
Ask questions about:

- Criminal Law
- Cyber Law
- Consumer Protection
- Digital Rights
- Online Fraud
- Legal Procedures

The assistant first searches the uploaded legal corpus and then uses corrective web search if necessary.
"""
)

with st.expander("💡 Example Questions"):

    st.markdown(
        """
- What is cyber terrorism under the IT Act?
- What are consumer rights under the Consumer Protection Act?
- What is a cognizable offence under BNSS?
- What penalties exist for data theft?
- What amendments were introduced to the IT Act in 2025?
"""
    )

@st.cache_resource
def load_rag():

    return AgenticRAG(LEGAL_DOCS)

try:

    rag_system = load_rag()

except Exception as e:

    st.error(f"System Initialization Failed: {e}")

    st.stop()

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a legal question..."
)

if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing legal documents and generating answer..."
        ):

            start = time.time()

            try:

                history_for_model = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.chat_history
                ]

                result = rag_system.ask(
                    question,
                    history_for_model,
                    st.session_state.mode
                )

                answer = result["answer"]

                source_type = result["source_type"]

                confidence = result["confidence"]

            except Exception as e:

                answer = f"Error: {e}"

                source_type = "system"

                confidence = "LOW"

            elapsed = time.time() - start

            if source_type == "vectorstore":

                st.success(
                    "📚 Source: Legal Corpus | 🟢 High Confidence"
                )

            elif source_type == "external":

                st.warning(
                    "🌐 Source: External Search | 🟡 Medium Confidence"
                )

            else:

                st.error(
                    "⚠️ System Response | 🔴 Low Confidence"
                )

            st.markdown(answer)

            st.caption(
                f"⏱️ Response Time: {elapsed:.2f} seconds"
            )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )