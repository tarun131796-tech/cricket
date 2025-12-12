from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from src.agent.tools import predict_match_outcome
import os

# Define State
class AgentState(TypedDict):
    messages: list[BaseMessage]
    match_details: dict
    prediction: dict
    explanation: str

# Define Nodes
def input_validation_node(state: AgentState):
    """Validates the input and extracts match details."""
    # Check if messages exist, if not, we rely on match_details which should be populated by the API caller
    if state.get('messages') and len(state['messages']) > 0:
        last_message = state['messages'][-1].content
    
    # In a real scenario, we might use an LLM here to extract entities if they are unstructured.
    # For now, we assume the input is already structured or passed in `match_details` if invoked via API.
    # If invoked via chat, we'd extract.
    
    # We will assume structured input is populated in `match_details` for the API use case.
    # If match_details is empty, we try to parse it (simplified).
    
    return state

def prediction_node(state: AgentState):
    """Calls the ML model prediction tool."""
    details = state.get("match_details", {})
    if not details:
         return {"prediction": {"error": "No match details provided"}}

    result = predict_match_outcome.invoke(details)
    return {"prediction": result}

def explanation_node(state: AgentState):
    """Generates an explanation using Gemini."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7) # Using 2.0 as requested/available, close to 2.5
    
    prediction = state.get("prediction", {})
    match_details = state.get("match_details", {})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cricket expert analyst. Explain the match prediction results."),
        ("human", "Match: {match_details}\nPrediction: {prediction}\n\nProvide a detailed reasoning for why this team might win based on typical cricket factors (venue, toss, team strength) and the provided probability.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"match_details": match_details, "prediction": prediction})
    
    return {"explanation": response.content}

# Build Graph
def build_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("validate", input_validation_node)
    workflow.add_node("predict", prediction_node)
    workflow.add_node("explain", explanation_node)
    
    workflow.set_entry_point("validate")
    workflow.add_edge("validate", "predict")
    workflow.add_edge("predict", "explain")
    workflow.add_edge("explain", END)
    
    return workflow.compile()

if __name__ == "__main__":
    # Test run (requires GOOGLE_API_KEY)
    pass
