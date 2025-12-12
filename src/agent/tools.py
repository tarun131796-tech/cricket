import joblib
import pandas as pd
from langchain.tools import tool

# Global variable to cache the model
_MODEL = None

def get_model():
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = joblib.load("models/best_model.pkl")
        except FileNotFoundError:
            return None
    return _MODEL

@tool
def predict_match_outcome(team1: str, team2: str, venue: str, toss_winner: str, toss_decision: str) -> dict:
    """
    Predicts the winner of a cricket match given the teams, venue, and toss details.
    
    Args:
        team1: Name of the first team.
        team2: Name of the second team.
        venue: The stadium where the match is played.
        toss_winner: The team that won the toss.
        toss_decision: The decision taken by the toss winner ('Bat' or 'Bowl').
        
    Returns:
        A dictionary containing the predicted winner and the win probability.
    """
    model = get_model()
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    input_data = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision
    }])
    
    try:
        # Predict probability for Team 1
        prob_team1 = model.predict_proba(input_data)[0][1]
        
        if prob_team1 > 0.5:
            winner = team1
            confidence = prob_team1
        else:
            winner = team2
            confidence = 1 - prob_team1
            
        return {
            "predicted_winner": winner,
            "win_probability": float(confidence),
            "team1_win_prob": float(prob_team1),
            "team2_win_prob": float(1 - prob_team1)
        }
    except Exception as e:
        return {"error": str(e)}
