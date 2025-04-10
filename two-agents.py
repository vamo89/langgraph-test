from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import TypedDict

# 1. Define the state
class ConversationState(TypedDict):
    input: str
    agent1_output: str
    agent2_output: str

# 2. Setup Ollama model
llm = ChatOllama(model="mistral")

# 3. First agent node
def agent1(state: ConversationState):
    response = llm.invoke([HumanMessage(content=f"Agent 1 received: {state['input']}. Please analyze it.")])
    return {"agent1_output": response.content}

# 4. Second agent node
def agent2(state: ConversationState):
    response = llm.invoke([HumanMessage(content=f"Agent 2 building on Agent 1's work: {state['agent1_output']}")])
    return {"agent2_output": response.content}

# 5. Build the graph
graph = StateGraph(state_schema=ConversationState)
graph.add_node("agent1", agent1)
graph.add_node("agent2", agent2)

# Define the flow
graph.add_edge("agent1", "agent2")  # agent1 ➔ agent2
graph.set_entry_point("agent1")

# 6. Compile
graph_executor = graph.compile()

# 7. Run
if __name__ == "__main__":
    result = graph_executor.invoke({"input": "Explain the importance of AI in modern medicine."})
    print("\nFinal Output:")
    print(result)
