from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import TypedDict

# 1. Define the State schema
class ConversationState(TypedDict):
    input: str
    output: str

# 2. Setup Ollama model
llm = ChatOllama(model="mistral")

# 3. Node function
def simple_agent(state: ConversationState):
    response = llm.invoke([HumanMessage(content=state["input"])])
    return {"output": response.content}

# 4. Build the graph
graph = StateGraph(state_schema=ConversationState)
graph.add_node("agent", simple_agent)
graph.set_entry_point("agent")

# 5. Compile
graph_executor = graph.compile()

# 6. Run
if __name__ == "__main__":
    result = graph_executor.invoke({"input": "Summarize the latest AI trends"})
    print("\nResult:")
    print(result)
