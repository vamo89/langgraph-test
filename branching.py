from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import TypedDict

# 1. Define State
class ConversationState(TypedDict):
    input: str
    agent1_output: str
    agent2_output: str
    agent3_output: str

# 2. Setup LLM
llm = ChatOllama(model="mistral")

# 3. Agent 1: Analyzes input
def agent1(state: ConversationState):
    response = llm.invoke([HumanMessage(content=f"Classify this input as either 'technical' or 'general': {state['input']}")])
    return {"agent1_output": response.content.lower()}

# 4. Agent 2: If the topic is technical
def agent2(state: ConversationState):
    response = llm.invoke([HumanMessage(content=f"Provide a detailed technical analysis of: {state['input']}")])
    return {"agent2_output": response.content}

# 5. Agent 3: If the topic is general
def agent3(state: ConversationState):
    response = llm.invoke([HumanMessage(content=f"Summarize this topic for a general audience: {state['input']}")])
    return {"agent3_output": response.content}

# 6. Conditional routing function
def route(state: ConversationState):
    if "technical" in state["agent1_output"]:
        return "agent2"
    else:
        return "agent3"

# 7. Build Graph
graph = StateGraph(state_schema=ConversationState)
graph.add_node("agent1", agent1)
graph.add_node("agent2", agent2)
graph.add_node("agent3", agent3)

graph.add_conditional_edges("agent1", route)   # After agent1, decide: agent2 or agent3
graph.set_entry_point("agent1")

# 8. Compile
graph_executor = graph.compile()

# 9. Run
if __name__ == "__main__":
    result = graph_executor.invoke({"input": "Explain the architecture of Transformer models in deep learning."})
    print("\nFinal Result:")
    print(result)
