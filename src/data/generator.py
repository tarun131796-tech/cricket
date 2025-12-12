import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_matches=1000):
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
    
    data = []
    
    for _ in range(num_matches):
        team1, team2 = random.sample(teams, 2)
        venue = random.choice(venues)
        toss_winner = random.choice([team1, team2])
        toss_decision = random.choice(["Bat", "Bowl"])
        
        # Simulate match outcome logic (simplified)
        # Team strength factor (randomized slightly)
        team_strength = {t: random.uniform(0.4, 0.9) for t in teams}
        
        # Home advantage (fake)
        home_advantage = 0.1 if "India" in [team1, team2] and "India" in venue else 0.0
        
        # Toss advantage
        toss_advantage = 0.05 if toss_winner == team1 and toss_decision == "Bat" else 0.0
        
        prob_team1 = 0.5 + (team_strength[team1] - team_strength[team2]) / 2 + home_advantage + toss_advantage
        prob_team1 = max(0.1, min(0.9, prob_team1))
        
        winner = team1 if random.random() < prob_team1 else team2
        
        # Additional features
        first_innings_score = int(random.normalvariate(280, 40)) if toss_decision == "Bat" else int(random.normalvariate(260, 40))
        wickets_lost = int(random.normalvariate(6, 2))
        wickets_lost = max(0, min(10, wickets_lost))

        
        data.append({
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "winner": winner
        })
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(2000)
    df.to_csv("data/matches.csv", index=False)
    print("Synthetic data generated at data/matches.csv")
