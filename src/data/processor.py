import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def load_and_process_data(filepath="data/matches.csv"):
    df = pd.read_csv(filepath)
    
    # Define features and target
    X = df.drop(columns=["winner"])
    y = df["winner"]
    
    # We need to ensure the target variable matches one of the teams. 
    # For binary classification (Team 1 wins vs Team 2 wins), it's tricky because teams vary per row.
    # Instead, we can predict if Team 1 wins.
    
    y_binary = (y == df["team1"]).astype(int) # 1 if Team 1 wins, 0 if Team 2 wins
    
    return X, y_binary

def get_preprocessor():
    categorical_features = ["team1", "team2", "venue", "toss_winner", "toss_decision"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

if __name__ == "__main__":
    X, y = load_and_process_data()
    print(f"Data loaded. Shape: {X.shape}")
