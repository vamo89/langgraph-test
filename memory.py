from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

# 1. Define State
class ConversationState(TypedDict):
    history: List[str]
    input: str
    agent1_output: str
    agent2_output: str

# 2. Setup LLM
llm = ChatOllama(model="mistral")

# 3. Agent 1: Responds based on input
def agent1(state: ConversationState):
    memory = "\n".join(state.get("history", []))
    prompt = f"Based on conversation history:\n{memory}\n\nAnswer the question: {state['input']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"agent1_output": response.content}

# 4. Agent 2: Builds on agent1 output + history
def agent2(state: ConversationState):
    memory = "\n".join(state.get("history", [])) + "\n" + state["agent1_output"]
    prompt = f"Building on the discussion so far:\n{memory}\n\nExpand and provide more detail."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"agent2_output": response.content}

# 5. Graph build
graph = StateGraph(state_schema=ConversationState)
graph.add_node("agent1", agent1)
graph.add_node("agent2", agent2)

# Update history after agent1 before agent2
graph.add_edge("agent1", "agent2")
graph.set_entry_point("agent1")

# 6. Compile
graph_executor = graph.compile()

# 7. Run
if __name__ == "__main__":
    result = graph_executor.invoke({
        "input": "What are the main challenges of using AI in healthcare?",
        "history": []
    })
    print("\nFinal Output:")
    print(result)
