from src.state.ragstate import RAGstate
from src.config.config import Config

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults


class Nodes:

    def __init__(self, retriever, llm):

        self.retriever = retriever

        self.llm = llm

        self.evaluator_llm = Config.get_evaluator_llm()

        self.tavily = TavilySearchResults(max_results=5)

    def rewrite_question(
        self,
        question: str,
        chat_history: list
    ) -> str:

        if not chat_history:

            return question

        recent_history = chat_history[-6:]

        history_text = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in recent_history
            ]
        )

        prompt = f"""
You are a query rewriting assistant.

Conversation History:

{history_text}

Latest User Question:

{question}

Rewrite the latest question into a standalone question.

The rewritten question must preserve the original meaning while incorporating any necessary context from the conversation history.

Return ONLY the rewritten question.
"""

        response = self.llm.invoke(prompt)

        rewritten_question = response.content.strip()

        print(f"\nOriginal Question: {question}")
        print(f"Rewritten Question: {rewritten_question}")

        return rewritten_question

    def evaluate_context(self, question: str, context: str) -> str:

        prompt = f"""
You are a retrieval evaluator.

Question:
{question}

Retrieved Context:
{context}

Can the question be answered accurately and completely using ONLY the retrieved context?

Return ONLY one word:

SUFFICIENT

or

INSUFFICIENT
"""

        response = self.evaluator_llm.invoke(prompt)

        result = response.content.strip().upper()

        print(f"Evaluator Output: {result}")

        if result.startswith("INSUFFICIENT"):
            return "INSUFFICIENT"

        if result.startswith("SUFFICIENT"):
            return "SUFFICIENT"

        return "INSUFFICIENT"

    def external_search(self, question: str) -> str:

        try:

            result = self.tavily.invoke(question)

            print("Tavily Search Triggered")

            return str(result)

        except Exception as e:

            return f"External search failed: {str(e)}"

    def generate_corrective_answer(
        self,
        question: str,
        external_context: str
    ) -> str:

        prompt = f"""
You are an AI legal assistant specialized in Indian laws.

QUESTION:
{question}

EXTERNAL INFORMATION:
{external_context}

Instructions:

1. Begin with:

Source: External Search

2. Explain that the answer was not found
in the uploaded legal corpus.

3. Use the external information to answer.

4. If the information comes from a proposed bill,
draft amendment, consultation paper, or news report,
explicitly mention this.

5. Provide a structured answer.

6. End with:

Confidence Level: Medium
"""

        response = self.llm.invoke(prompt)

        return response.content

    def generate_rag_answer(
        self,
        question: str,
        context: str
    ) -> str:

        prompt = f"""
You are an AI legal assistant specialized in Indian laws.

Use ONLY the legal context below.

LEGAL CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = self.llm.invoke(prompt)

        return response.content

    def generate_ans(self, state: RAGstate) -> RAGstate:

        try:

            rewritten_question = self.rewrite_question(
                state.question,
                state.chat_history
            )

            docs: list[Document] = self.retriever.retrieve(
                rewritten_question
            )

            if not docs:

                print("No documents retrieved. Switching to Tavily.")

                external_context = self.external_search(
                    rewritten_question
                )

                answer = self.generate_corrective_answer(
                    rewritten_question,
                    external_context
                )

                return RAGstate(
                    question=state.question,
                    chat_history=state.chat_history,
                    retrieved_docs=[],
                    answer=answer,
                    context_quality="INSUFFICIENT",
                    source_type="external",
                    external_context=external_context
                )

            context = []

            for i, d in enumerate(docs, start=1):

                context.append(
                    f"[Document {i}]\n{d.page_content}"
                )

            merged_context = "\n\n".join(context)

            context_quality = self.evaluate_context(
                rewritten_question,
                merged_context
            )

            print(f"Context Quality Decision: {context_quality}")

            if context_quality == "INSUFFICIENT":

                print("Using Tavily corrective search...")

                external_context = self.external_search(
                    rewritten_question
                )

                answer = self.generate_corrective_answer(
                    rewritten_question,
                    external_context
                )

                return RAGstate(
                    question=state.question,
                    chat_history=state.chat_history,
                    retrieved_docs=docs,
                    answer=answer,
                    context_quality="INSUFFICIENT",
                    source_type="external",
                    confidence="MEDIUM",
                    external_context=external_context
                )

            answer = self.generate_rag_answer(
                rewritten_question,
                merged_context
            )

            return RAGstate(
                question=state.question,
                chat_history=state.chat_history,
                retrieved_docs=docs,
                answer=answer,
                context_quality="SUFFICIENT",
                source_type="vectorstore",
                confidence="HIGH",
                external_context=""
            )

        except Exception as e:

            return RAGstate(
                question=state.question,
                chat_history=state.chat_history,
                retrieved_docs=[],
                answer=f"Error: {str(e)}",
                context_quality="ERROR",
                source_type="system",
                confidence="LOW",
                external_context=""
            )