"""Graph builder for LangGraph workflow"""

from langgraph.graph import StateGraph, END
from src.state.ragstate import RAGstate
from src.nodes.node import Nodes  # Assuming Nodes class is in reactnode.py

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = Nodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGstate)
        
        # Add nodes
        builder.add_node("agent", self.nodes.generate_ans)
        
        # Set entry point
        builder.set_entry_point("agent")
        
        # Add edge to end
        builder.add_edge("agent", END)
        
        # Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGstate(question=question)
        return self.graph.invoke(initial_state)   