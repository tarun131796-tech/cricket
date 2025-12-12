# Cricket Match Win Predictor System

This project is an end-to-end Machine Learning system to predict cricket match outcomes. It uses **XGBoost** for prediction, **Gemini 2.5** (via LangChain) for reasoning, **LangGraph** for workflow orchestration, and **FastAPI/Streamlit** for the application layer.

## Project Structure

```
cricket_predictor/
├── data/                  # Generated data and raw CSVs
├── models/                # Trained ML models
├── src/
│   ├── data/
│   │   ├── generator.py   # Generates synthetic data
│   │   └── processor.py   # Data preprocessing pipeline
│   ├── model/
│   │   └── train.py       # Model training (XGBoost/RF)
│   ├── agent/
│   │   ├── tools.py       # LangChain tools (ML Wrapper)
│   │   └── graph.py       # LangGraph agent definition
│   ├── server/
│   │   └── app.py         # FastAPI + LangServe backend
│   └── frontend/
│       └── app.py         # Streamlit UI
├── requirements.txt       # Dependencies
└── README.md              # Instructions
```

## Setup Instructions

### 1. Environment Setup

It is recommended to use a virtual environment.

```bash
# Create venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set API Keys

Create a `.env` file in the root directory and add your Google API Key for Gemini.

```
GOOGLE_API_KEY=your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here # Optional for tracing
```

### 3. Data Generation & Model Training

Before running the app, you need data and a trained model.

```bash
# 1. Generate synthetic data
python src/data/generator.py

# 2. Train the model
python src/model/train.py
```

### 4. Running the Application

You need to run the Backend and Frontend in separate terminals.

**Terminal 1: Backend (FastAPI)**
```bash
python src/server/app.py
```
*Server will run at http://0.0.0.0:8000*

**Terminal 2: Frontend (Streamlit)**
```bash
streamlit run src/frontend/app.py
```
*Frontend will run at http://localhost:8501*

## Features

1.  **Synthetic Data Pipeline**: Generates realistic cricket match scenarios.
2.  **ML Engine**: Compares Random Forest and XGBoost, selects the best one.
3.  **LangGraph Agent**: Validates input -> Predicts -> Explains.
4.  **Generative AI**: Uses Gemini to explain *why* a team is predicted to win based on match context.
5.  **Observability**: Integrated with LangSmith (configurable via env vars).

## How to Extend

*   **Data**: Replace `generator.py` with a script to scrape espncricinfo or load Kaggle datasets.
*   **Model**: Add hyperparameter tuning (GridSearchCV) in `train.py`.
*   **Agent**: Add more tools (e.g., `get_player_stats`, `check_weather`) to the LangGraph for richer analysis.
