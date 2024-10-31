import pandas as pd
import lightgbm as lgb
import sys
import yaml
import pathlib
import pickle
import logging
import numpy as np
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_split_data(dataset_path: str) -> tuple:
    try:
        logging.info(f"Loading dataset from {dataset_path}.")
        dataset = pd.read_csv(dataset_path)
        xtrain = dataset.iloc[:, 0:-1]
        ytrain = dataset.iloc[:, -1]
        logging.info("Data loaded and split successfully.")
        return xtrain, ytrain
    except Exception as e:
        logging.error(f"Error loading or splitting data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, parameters: dict) -> lgb.LGBMClassifier:
    try:
        logging.info("Training LGBMClassifier model.")
        model = lgb.LGBMClassifier(
            objective= 'multiclass',
            num_class= 3,
            metric= "multi_logloss",
            is_unbalance= True,
            class_weight= "balanced",
            reg_alpha= 0.1,  # L1 regularization
            reg_lambda= 0.1,  # L2 regularization
            learning_rate= parameters['learning_rate'],
            max_depth= parameters['max_depth'],
            n_estimators= parameters['n_estimators']
        )
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training the model: {e}")
        raise

def save_model(model, file_path: str):
    try:
        pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
        with open(file_path + "/model.pkl", 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved successfully at {file_path}/model.pkl.")
    except Exception as e:
        logging.error(f"Error saving the model: {e}")
        raise

def main():
    try:
        current_dir = pathlib.Path(__file__)
        home_dir = current_dir.parent.parent.parent

        path = sys.argv[1]
        model_saving_path = home_dir.as_posix() + "/models"
        processed_datasets_path = home_dir.as_posix() + path + "/processed_traindata.csv"

        params_location = home_dir.as_posix() + '/params.yaml'
        parameters = yaml.safe_load(open(params_location))["train_model"]

        x, y = load_and_split_data(processed_datasets_path)
        model = train_model(x, y, parameters)
        
        save_model(model, model_saving_path)

        logging.info("Main function completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
