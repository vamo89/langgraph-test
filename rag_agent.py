from typing import Dict, List, TypedDict, Annotated, Tuple
from langgraph.graph import Graph, StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Define the state
class AgentState(TypedDict):
    messages: List[Dict]
    current_message: str

# Initialize components
def initialize_components():
    # Initialize the LLM
    llm = OllamaLLM(model="llama2")
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="llama2")
    
    # Load and process documents
    loader = DirectoryLoader("data", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    # Create RAG prompt
    template = """Answer the question based on the following context:
    
    Context: {context}
    
    Question: {question}
    
    Answer: Let me think step by step."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Define the agent nodes
def retrieve_and_answer(state: AgentState) -> AgentState:
    rag_chain = initialize_components()
    
    # Get the current message
    current_message = state["current_message"]
    
    # Get the answer
    answer = rag_chain.invoke(current_message)
    
    # Update the state
    state["messages"].append(
        {"role": "assistant", "content": answer}
    )
    
    return state

# Create the graph
def create_graph() -> Graph:
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_and_answer", retrieve_and_answer)
    
    # Set the entry point
    workflow.set_entry_point("retrieve_and_answer")
    
    # Add conditional edges
    workflow.add_edge("retrieve_and_answer", END)
    
    # Compile the graph
    return workflow.compile()

# Main function
def main():
    # Create the graph
    graph = create_graph()
    
    # Initialize the state
    initial_state = {
        "messages": [],
        "current_message": "What is machine learning?"
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    # Print the result
    print("\nQuestion:", initial_state["current_message"])
    print("\nAnswer:", result["messages"][-1]["content"])

if __name__ == "__main__":
    main() 