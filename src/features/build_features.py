import pandas as pd
import sys
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
import pathlib
import logging
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, no_of_features: int, ngram_range: tuple) -> tuple:
    try:
        logging.info("Starting TF-IDF transformation.")

        vectorizer = TfidfVectorizer(max_features=no_of_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        X_test = test_data['clean_comment'].values
        y_test = test_data['category'].values

        logging.info("Fitting TF-IDF Vectorizer to training data.")
        X_train_bow = vectorizer.fit_transform(X_train)
        logging.info("Transforming test data.")
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['category'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['category'] = y_test

        logging.info("TF-IDF transformation completed successfully.")
        return train_df, test_df, vectorizer
    except Exception as e:
        logging.error(f"Error during TF-IDF transformation: {e}")
        raise

def save_data(traindata: pd.DataFrame, testdata: pd.DataFrame, path: str) -> None:
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        traindata.to_csv(path + "/processed_traindata.csv", index=False, header=True)
        testdata.to_csv(path + "/processed_testdata.csv", index=False, header=True)
        logging.info("Processed data saved successfully.")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise

def save_vectorizer(vectorizer: TfidfVectorizer, path_location: str) -> None:
    try:
        pathlib.Path(path_location).mkdir(parents=True, exist_ok=True)
        with open(path_location + "/vectorizer.pkl", 'wb') as file:
            pickle.dump(vectorizer, file)
        logging.info("Vectorizer saved successfully at models/vectorizer.pkl.")
    except Exception as e:
        logging.error(f"Error saving vectorizer: {e}")
        raise


def main():
    try:
        current_dir = pathlib.Path(__file__)
        home_dir = current_dir.parent.parent.parent

        path = sys.argv[1]
        processed_data_saving_path = home_dir.as_posix() + path + "/external"
        preprocess_datasets_path = home_dir.as_posix() + path + "/interim"
        vector_saving_path = home_dir.as_posix() + "/models"
        
        params_location = home_dir.as_posix() + '/params.yaml'
        parameters = yaml.safe_load(open(params_location))["build_features"]

        logging.info("Loading preprocessed datasets.")
        train_df = pd.read_csv(preprocess_datasets_path + "/preprocess_traindata.csv")
        test_df = pd.read_csv(preprocess_datasets_path + "/preprocess_testdata.csv")
      
        train_df.fillna('', inplace=True)
        test_df.fillna('', inplace=True)

        ngram_range = tuple(parameters['ngram_range'])

        process_train_df, process_test_df, vectorizer = apply_tfidf(train_df, test_df, parameters["max_features"], ngram_range)
        save_data(process_train_df, process_test_df, processed_data_saving_path)
        save_vectorizer(vectorizer, vector_saving_path)

        logging.info("Main function completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()

