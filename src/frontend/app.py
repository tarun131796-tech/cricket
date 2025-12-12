import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Cricket Win Predictor", layout="wide")

st.title("🏏 AI Cricket Match Win Predictor")
st.markdown("Powered by **Gemini 2.5**, **LangGraph**, and **XGBoost**")

# Sidebar for inputs
with st.sidebar:
    st.header("Match Details")
    
    teams = [
        "India", "Australia", "England", "South Africa", "New Zealand", 
        "Pakistan", "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan"
    ]
    
    venues = [
        "Wankhede Stadium, Mumbai", 
        "Melbourne Cricket Ground", 
        "Lord's, London", 
        "Eden Gardens, Kolkata", 
        "Sydney Cricket Ground", 
        "The Oval, London", 
        "Newlands, Cape Town", 
        "Dubai International Cricket Stadium",
        "Narendra Modi Stadium, Ahmedabad"
    ]
    
    team1 = st.selectbox("Team 1", teams, index=0)
    team2 = st.selectbox("Team 2", teams, index=1)
    venue = st.selectbox("Venue", venues)
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["Bat", "Bowl"])
    
    predict_btn = st.button("Predict Winner")

if predict_btn:
    if team1 == team2:
        st.error("Team 1 and Team 2 must be different.")
    else:
        with st.spinner("Analyzing match data and generating explanation..."):
            try:
                # Call the FastAPI backend
                # Assuming running locally on port 8000
                api_url = "http://localhost:8000/predict_custom"
                payload = {
                    "team1": team1,
                    "team2": team2,
                    "venue": venue,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision
                }
                
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    prediction = data.get("prediction", {})
                    explanation = data.get("explanation", "")
                    
                    if "error" in prediction:
                        st.error(f"Error: {prediction['error']}")
                    else:
                        # Display Results
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("Prediction")
                            winner = prediction.get('predicted_winner')
                            prob = prediction.get('win_probability', 0)
                            
                            st.metric("Predicted Winner", winner)
                            st.metric("Confidence", f"{prob:.1%}")
                            
                            # Chart
                            chart_data = pd.DataFrame({
                                "Team": [team1, team2],
                                "Probability": [prediction['team1_win_prob'], prediction['team2_win_prob']]
                            })
                            
                            fig, ax = plt.subplots()
                            sns.barplot(x="Team", y="Probability", data=chart_data, ax=ax, palette=["#1f77b4", "#ff7f0e"])
                            ax.set_ylim(0, 1)
                            st.pyplot(fig)

                        with col2:
                            st.subheader("Analysis & Reasoning")
                            st.markdown(explanation)
                            
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
                st.info("Make sure the FastAPI server is running: `python src/server/app.py`")
