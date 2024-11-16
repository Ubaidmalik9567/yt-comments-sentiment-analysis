import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import os
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_mlflow_tracking():
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Ubaidmalik9567"
    repo_name = "yt-comments-sentiment-analysis"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def get_latest_model_version(model_name, stage):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next(
        (v for v in model_versions if v.current_stage == stage), None
    )

    if not latest_version_info:
        raise Exception(f"No model found in the '{stage}' stage.")

    logging.info(
        f"Model '{model_name}' | Version: {latest_version_info.version} | ID: {latest_version_info.run_id} | Stage: '{stage}'"
    )
    return latest_version_info.run_id, latest_version_info.version

def download_artifacts(run_id, download_path):
    client = MlflowClient()
    os.makedirs(download_path, exist_ok=True)
    client.download_artifacts(run_id, "", download_path)
    logging.info(f"Artifacts downloaded to: {download_path}")

    for root, dirs, files in os.walk(download_path):
        for file in files:
            logging.info(f"Found file: {os.path.join(root, file)}")

def load_model_from_artifacts(download_path):
    model_pkl_path = None
    for root, dirs, files in os.walk(download_path):
        if 'model.pkl' in files:
            model_pkl_path = os.path.join(root, 'model.pkl')
            break

    if model_pkl_path:
        logging.info(f"Found model.pkl at: {model_pkl_path}")
        with open(model_pkl_path, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        return model
    else:
        logging.error("model.pkl not found in downloaded artifacts.")
        return None

@pytest.mark.parametrize("model_name, stage", [
    ("save_model", "staging"),
])
def test_model_loading_process(model_name, stage):
    try:
        setup_mlflow_tracking()
        
        # Step 1: Get the latest model version
        run_id, version = get_latest_model_version(model_name, stage)
        assert run_id is not None, "Run ID should not be None"

        # Step 2: Download artifacts
        download_path = "artifacts"
        download_artifacts(run_id, download_path)

        # Step 3: Load the model
        model = load_model_from_artifacts(download_path)
        assert model is not None, "Model should load successfully"

        logging.info(
            f"Model '{model_name}' | Version: {version} | ID: {run_id} | Stage: '{stage}' successfully loaded and verified."
        )

    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")
