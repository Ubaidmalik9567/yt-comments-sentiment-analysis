import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import yaml
import sys
import pathlib
from sklearn.model_selection import train_test_split
import logging
import nltk

nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocessing(raw_dataset_path: str, split_size: float, seed: float):
    try:
        # Load dataset
        df = pd.read_csv(raw_dataset_path)
    
        df.drop(index=0, inplace=True)  # Dropping the first row
        df.reset_index(drop=True, inplace=True)  # Resetting index

        logging.info("Loaded raw dataset successfully.")
    except Exception as e:
        logging.error(f"Failed to load raw dataset: {e}")
        raise

    try:
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[~(df['clean_comment'].str.strip() == '')]  
        df['clean_comment'] = df['clean_comment'].str.replace('\n', ' ', regex=True)

        logging.info("Preprocessing completed: cleaned dataset and removed unnecessary rows.")
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

    try:
        train_data, test_data = train_test_split(df, test_size=split_size, random_state=seed)
        logging.info("Split data into training and test sets.")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        raise

def preprocess_comment(comment):

    comment = comment.lower()
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment


def normalize_text(dataset):
    try:
        dataset["clean_comment"] = dataset["clean_comment"].apply(preprocess_comment)
        logging.info("Text normalization completed successfully.")

        return dataset
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        raise

def save_data(traindata: pd.DataFrame, testdata: pd.DataFrame, path: str) -> None:
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        traindata.to_csv(path + "/preprocess_traindata.csv", index=False, header=True)
        testdata.to_csv(path + "/preprocess_testdata.csv", index=False, header=True)
        logging.info("Saved processed data successfully.")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")
        raise

def main():
    try:
        current_dir = pathlib.Path(__file__)
        home_dir = current_dir.parent.parent.parent

        path = sys.argv[1]
        predata_save_path = home_dir.as_posix() + path + "/interim"
        raw_dataset_path = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv" # home_dir.as_posix() + path + "/raw/rawdata.csv"
        
        params_location = home_dir.as_posix() + '/params.yaml'
        parameters = yaml.safe_load(open(params_location))["data_transformation"]

        train_data, test_data = preprocessing(raw_dataset_path, parameters["split_dataset_size"], parameters["seed"])
        pretrain_data = normalize_text(train_data)
        pretest_data = normalize_text(test_data)
        save_data(pretrain_data, pretest_data, predata_save_path)
        logging.info("Data processing pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()

