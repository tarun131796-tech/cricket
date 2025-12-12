import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import sys
import os

# Add src to path
sys.path.append(os.getcwd())
from src.data.processor import load_and_process_data, get_preprocessor

def train_models():
    X, y = load_and_process_data("data/matches.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = get_preprocessor()
    
    # Model 1: Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    
    print("Random Forest Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, rf_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    
    # Model 2: XGBoost
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    xgb_pipeline.fit(X_train, y_train)
    xgb_preds = xgb_pipeline.predict(X_test)
    xgb_probs = xgb_pipeline.predict_proba(X_test)[:, 1]
    
    print("\nXGBoost Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    print(f"F1 Score: {f1_score(y_test, xgb_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    
    # Select best model (Simple logic: based on Accuracy)
    rf_acc = accuracy_score(y_test, rf_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    
    if xgb_acc > rf_acc:
        best_model = xgb_pipeline
        print("\nBest Model: XGBoost")
    else:
        best_model = rf_pipeline
        print("\nBest Model: Random Forest")
        
    joblib.dump(best_model, "models/best_model.pkl")
    print("Model saved to models/best_model.pkl")

if __name__ == "__main__":
    train_models()
