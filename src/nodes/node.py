from src.state.ragstate import RAGstate
from src.config.config import Config

from langchain_core.documents import Document


class Nodes:

    def __init__(self, retriever, llm):

        self.retriever = retriever

        self.llm = llm

        self.evaluator_llm = Config.get_evaluator_llm()

    def evaluate_context(self, question: str, context: str) -> str:

        prompt = f"""
You are a retrieval evaluator.

Question:
{question}

Retrieved Context:
{context}

Determine whether the retrieved context contains sufficient information
to answer the question accurately.

Return ONLY one word:

SUFFICIENT

or

INSUFFICIENT
"""

        response = self.evaluator_llm.invoke(prompt)

        result = response.content.strip().upper()

        if "SUFFICIENT" in result:
            return "SUFFICIENT"

        return "INSUFFICIENT"

    def generate_ans(self, state: RAGstate) -> RAGstate:

        try:

            docs: list[Document] = self.retriever.retrieve(state.question)

            if not docs:

                return RAGstate(
                    question=state.question,
                    retrieved_docs=[],
                    answer="No relevant legal documents found.",
                    context_quality="INSUFFICIENT",
                    source_type="vectorstore"
                )

            context = []

            for i, d in enumerate(docs, start=1):

                context.append(
                    f"[Document {i}]\n{d.page_content}"
                )

            merged_context = "\n\n".join(context)

            context_quality = self.evaluate_context(
                state.question,
                merged_context
            )

            print(f"Context Quality: {context_quality}")

            prompt = f"""
You are an AI legal assistant specialized in Indian laws.

Retrieved Context Quality:
{context_quality}

Use ONLY the legal context provided below.

If context quality is INSUFFICIENT, clearly mention that the uploaded legal corpus may not contain enough information to fully answer the question.

LEGAL CONTEXT:
{merged_context}

QUESTION:
{state.question}

ANSWER:
"""

            response = self.llm.invoke(prompt)

            answer = response.content

            return RAGstate(
                question=state.question,
                retrieved_docs=docs,
                answer=answer,
                context_quality=context_quality,
                source_type="vectorstore"
            )

        except Exception as e:

            return RAGstate(
                question=state.question,
                retrieved_docs=[],
                answer=f"Error: {str(e)}",
                context_quality="ERROR",
                source_type="system"
            )