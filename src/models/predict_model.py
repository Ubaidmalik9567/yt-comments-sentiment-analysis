import pandas as pd
import sys
import pathlib
import pickle
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import json
import logging
import yaml
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(dataset_path: str) -> tuple:
    try:
        logging.info(f"Loading dataset from {dataset_path}.")
        dataset = pd.read_csv(dataset_path)
        xtest = dataset.iloc[:, 0:-1]
        ytest = dataset.iloc[:, -1]
        logging.info("Data loaded and split successfully.")
        return xtest, ytest
    except Exception as e:
        logging.error(f"Error loading or splitting data: {e}")
        raise

def load_save_model(file_path: str):
    try:
        logging.info(f"Loading model from {file_path}.")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        logging.info("Evaluating model performance.")
        y_pred = model.predict(X_test)
        # y_pred_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
     
        logging.info("Model evaluation completed successfully.")
        return report, cm
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving metrics to {file_path}/metrics.json.")
        with open(file_path + "/metrics.json", 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved successfully.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")
        raise

def log_confusion_matrix(cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = f'confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id, model_path, file_path) -> None: # that info use for model registry
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)

def random_sample_csv(csv_path, num_samples):
    df = pd.read_csv(csv_path)
    # Perform random sampling
    sampled_df = df.sample(n=num_samples, random_state=1)
    return sampled_df

def main():
    
    # Set up MLflow tracking URI
    # mlflow.set_tracking_uri("http://ec2-16-171-19-90.eu-north-1.compute.amazonaws.com:5000/")
    
    dagshub.init(repo_owner='Ubaidmalik9567', repo_name='yt-comments-sentiment-analysis', mlflow=True)
    mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/yt-comments-sentiment-analysis.mlflow")


    mlflow.set_experiment("dvc-pipeline-info")  # Set up MLflow experiment
    with mlflow.start_run(run_name="pred2prod_files-run") as run:  # Start MLflow run
        
        try:
            current_dir = pathlib.Path(__file__)
            home_dir = current_dir.parent.parent.parent

            path = sys.argv[1]
            save_metrics_location = home_dir.as_posix() + "/reports"
            processed_datasets_path = home_dir.as_posix() + path + "/processed_testdata.csv"
            trained_model_path = home_dir.as_posix() + "/models/model.pkl"
            # forsample_csv_path = processed_datasets_path

            # # Perform random sampling
            # # sampled_data = random_sample_csv(forsample_csv_path, num_samples=200)
            # sampled_data_path = home_dir.as_posix() + "/reports/sampled_data.csv"
            # # sampled_data.to_csv(sampled_data_path, index=False)
            # mlflow.log_artifact(sampled_data_path)

            x, y = load_and_split_data(processed_datasets_path)
            model = load_save_model(trained_model_path)

            report, cm = evaluate_model(model, x, y)
            save_metrics(report, save_metrics_location)

            with open("params.yaml", "r") as file:
                params = yaml.safe_load(file)

            # Log parameters from params.yaml
            for param, value in params.items():
                for key, value in value.items():
                    mlflow.log_param(f'{param}_{key}', value)
                    
            # Log all model parameters to MLflow
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)
            
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact("models/vectorizer.pkl")

            save_model_info(run.info.run_id, "models", 'reports/model_experiment_info.json')  # Save model info
            
            # Log the metrics, model info file to MLflow
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/model_experiment_info.json')

            # Evaluate model and get metrics
            report, cm = evaluate_model(model, x, y)

            # Log classification report metrics for the test data
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    mlflow.log_metrics({
                        f"test_{label}_precision": metrics['precision'],
                        f"test_{label}_recall": metrics['recall'],
                        f"test_{label}_f1-score": metrics['f1-score']
                    })

            # Log confusion matrix
            log_confusion_matrix(cm, "Test Data")

            # Add important tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")

            logging.info("Main function completed successfully.")
        except Exception as e:
            logging.error(f"Error in main function: {e}")
            raise

if __name__ == "__main__":
    main()
