from src.state.ragstate import RAGstate

from langchain_core.documents import Document


class Nodes():

    def __init__(self,retriever,llm):

        self.retriever=retriever
        self.llm=llm

    def generate_ans(self,state:RAGstate)->RAGstate:

        try:

            docs:list[Document]=self.retriever.retrieve(state.question)

            if not docs:

                answer="No relevant legal documents found."

                return RAGstate(
                    question=state.question,
                    retrieved_docs=[],
                    answer=answer
                )

            context=[]

            for i,d in enumerate(docs,start=1):

                context.append(
                    f"[Document {i}]\n{d.page_content}"
                )

            merged_context="\n\n".join(context)

            prompt=f"""
You are an AI legal assistant specialized in Indian laws.

Use ONLY the provided legal context to answer the question.

If the answer is not present in the context, clearly say:
'I could not find the answer in the legal documents.'

LEGAL CONTEXT:
{merged_context}

QUESTION:
{state.question}

ANSWER:
"""

            response=self.llm.invoke(prompt)

            answer=response.content

            return RAGstate(
                question=state.question,
                retrieved_docs=docs,
                answer=answer
            )

        except Exception as e:

            return RAGstate(
                question=state.question,
                retrieved_docs=[],
                answer=f"Error: {str(e)}"
            )