# Register model
import json
import mlflow
import os
import dagshub
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up MLflow tracking URI
# mlflow.set_tracking_uri("http://ec2-16-171-19-90.eu-north-1.compute.amazonaws.com:5000/")

dagshub.init(repo_owner='Ubaidmalik9567', repo_name='yt-comments-sentiment-analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/yt-comments-sentiment-analysis.mlflow")


def load_model_info(file_path):
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info(f"Model information loaded from {file_path}.")
        return model_info
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        raise


def register_model(model_name, model_info):
    #Register the model with MLflow and transition its stage to Staging.
    try:
        logging.info(f"Registering model: {model_name}.")
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)
        logging.info(f"Model registered with version: {model_version.version}.")
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logging.info(f"Model {model_name} transitioned to Staging stage.")
    except Exception as e:
        logging.error(f"Failed to register model {model_name}: {e}")
        raise

def main():
    try:
        model_info_path = 'reports/model_experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = "save_model"
        
        register_model(model_name, model_info)
        logging.info("Model registration process completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == '__main__':
    main()
