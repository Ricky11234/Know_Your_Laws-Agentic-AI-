import streamlit as st
import time

from main import AgenticRAG

st.set_page_config(
    page_title="Know Your Laws - Agentic AI",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Know Your Laws - Agentic AI")
st.markdown("Ask questions about Indian laws, cyber laws, and consumer protection laws.")

LEGAL_DOCS=[
    "data/legal_pdfs/Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf",
    "data/legal_pdfs/it_act_2000_updated.pdf",
    "data/legal_pdfs/CP Act 2019_1732700731.pdf"
]

@st.cache_resource
def load_rag():

    rag=AgenticRAG(LEGAL_DOCS)

    return rag

try:

    rag_system=load_rag()

    st.success("✅ Legal RAG System Loaded Successfully")

except Exception as e:

    st.error(f"System Initialization Failed: {e}")

    st.stop()

if "chat_history" not in st.session_state:

    st.session_state.chat_history=[]

question=st.chat_input("Ask your legal question here...")

if question:

    st.session_state.chat_history.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Generating Answer..."):

            start=time.time()

            try:

                answer=rag_system.ask(question)

            except Exception as e:

                answer=f"Error: {e}"

            elapsed=time.time()-start

            st.markdown(answer)

            st.caption(f"⏱️ Response Time: {elapsed:.2f} seconds")

    st.session_state.chat_history.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])