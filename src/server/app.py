from fastapi import FastAPI
from langserve import add_routes
from src.agent.graph import build_agent
from pydantic import BaseModel, Field
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Cricket Win Predictor Agent",
    version="1.0",
    description="An AI agent that predicts cricket match outcomes and explains them."
)

# Define Input Schema for the API
class MatchInput(BaseModel):
    team1: str = Field(..., description="Name of Team 1")
    team2: str = Field(..., description="Name of Team 2")
    venue: str = Field(..., description="Venue of the match")
    toss_winner: str = Field(..., description="Winner of the toss")
    toss_decision: str = Field(..., description="Toss decision (Bat/Bowl)")

# Wrapper to adapt API input to AgentState
def input_adapter(input_data: MatchInput):
    return {
        "messages": [],
        "match_details": input_data.dict(),
        "prediction": {},
        "explanation": ""
    }

# Create the graph runnable
agent_graph = build_agent()

# Add LangServe routes
add_routes(
    app,
    agent_graph,
    path="/agent",
)

@app.post("/predict_custom")
async def predict_custom(data: MatchInput):
    """Simple endpoint wrapper if LangServe is too complex for the frontend to consume directly."""
    state = {
        "messages": [],
        "match_details": data.dict(),
        "prediction": {},
        "explanation": ""
    }
    # Using ainvoke with the state
    result = await agent_graph.ainvoke(state)
    return {
        "prediction": result["prediction"],
        "explanation": result["explanation"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
