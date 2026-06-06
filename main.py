"""Main application for Agentic RAG"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.docu import docuprocessor
from src.vectorstore.faiss import Retriever
from src.graph_builder.g_builder import GraphBuilder


class AgenticRAG:

    def __init__(self, srcs: list[str]):

        print("Initializing Agentic RAG...")

        self.srcs = srcs

        self.processor = docuprocessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        self.retriever = Retriever()

        if self.retriever.index_exists():

            print("Loading existing FAISS index...")

            self.retriever.load_retriever()

        else:

            print("Processing legal documents...")

            docs = self.processor.process(srcs)

            print(f"Created {len(docs)} chunks")

            self.retriever.create_retriever(docs)

        self.llm = Config.get_llm()

        self.graph_builder = GraphBuilder(
            retriever=self.retriever,
            llm=self.llm
        )

        self.graph_builder.build()

        print("System Ready")

    def ask(self, question: str):

        result = self.graph_builder.run(question)

        answer = result["answer"]

        print("\nQuestion:")
        print(question)

        print("\nAnswer:")
        print(answer)

        return answer

    def interactive_mode(self):

        print("\nInteractive Mode Started")
        print("Type 'quit' to exit\n")

        while True:

            question = input("Enter Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye")
                break

            if question:

                self.ask(question)

                print("-" * 80)


def main():

    sources = [
        "data/legal_pdfs/Bharatiya_Nagarik_Suraksha_Sanhita,_2023.pdf",
        "data/legal_pdfs/it_act_2000_updated.pdf",
        "data/legal_pdfs/CP Act 2019_1732700731.pdf"
    ]

    rag = AgenticRAG(sources)

    rag.interactive_mode()


if __name__ == "__main__":
    main()