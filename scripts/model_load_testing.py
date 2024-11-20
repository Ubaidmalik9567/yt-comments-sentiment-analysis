import os
import mlflow
import pytest
from mlflow.tracking import MlflowClient
import dagshub

# Set your remote tracking URI
@pytest.fixture(scope="module")
def setup_environment():
    # Initialize Dagshub
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

    model_name = "save_model"
    stage = "staging"  # Change stage if needed

    # Get the latest model version from the specified stage
    run_id = get_latest_model_version(model_name, stage)
    if not run_id:
        pytest.fail(f"No model found in the '{stage}' stage.")

    # Print details about the model
    print(f"Model Name: {model_name}")
    print(f"Stage: {stage}")
    print(f"Run ID: {run_id}")

    return run_id, stage


def get_latest_model_version(model_name, stage="staging"):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next(
        (v for v in model_versions if v.current_stage == stage), None
    )
    return latest_version_info.run_id if latest_version_info else None


@pytest.mark.usefixtures("setup_environment")
def test_load_latest_staging_model(setup_environment):
    run_id, stage = setup_environment
    assert run_id, "Run ID should not be None"
    print(f"Successfully fetched model with Run ID: {run_id} in stage: {stage}")

    client = MlflowClient()

    try:
        # Fetch the latest version of the model from the staging stage
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_uri}' loaded successfully from '{stage}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
