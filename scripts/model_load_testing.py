import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
import logging
import os

# Set up Dagshub authentication
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set your remote Dagshub tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "Ubaidmalik9567"
repo_name = "yt-comments-sentiment-analysis"  # Replace with your actual repository name
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def get_latest_model_run_id(model_name, stage="staging"):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next((v for v in model_versions if v.current_stage == stage), None)
    return latest_version_info.run_id if latest_version_info else None

def load_model_and_vectorizer(model_name, stage="staging"):
    run_id = get_latest_model_run_id(model_name, stage)
    if not run_id:
        raise Exception(f"No model found in the '{stage}' stage.")
    
    # Load the model directly from MLflow (Dagshub)
    model_uri = f"runs:/{run_id}/model/model.pkl"
    model_path = mlflow.artifacts.download_artifacts(model_uri)
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")

    # Load the vectorizer from the MLflow (Dagshub) artifacts
    vectorizer_uri = f"runs:/{run_id}/vectorizer.pkl"
    vectorizer_path = mlflow.artifacts.download_artifacts(vectorizer_uri)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Vectorizer loaded successfully.")

    return model, vectorizer

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("save_model", "staging", "vectorizer.pkl"),  # Replace with your actual model name and vectorizer path
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    try:
        # Load the model and vectorizer
        model, vectorizer = load_model_and_vectorizer(model_name, stage)

        # Create a dummy input for the model
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())  # <-- Use correct feature names

        # Predict using the model
        prediction = model.predict(input_df)

        # Verify the input shape matches the vectorizer's feature output
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), "Input feature count mismatch"

        # Verify the output shape (assuming binary classification with a single output)
        assert len(prediction) == input_df.shape[0], "Output row count mismatch"

        print(f"Model '{model_name}' version successfully processed the dummy input.")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {e}")
