from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import logging
import dagshub

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

dagshub.init(repo_owner='Ubaidmalik9567', repo_name='yt-comments-sentiment-analysis', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ubaidmalik9567/yt-comments-sentiment-analysis.mlflow")

# Define a default route for the root URL
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the sentiment analysis API. Use the /predict endpoint to analyze comments."})

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def get_latest_model_run_id(model_name, stage="Production"):
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    latest_version_info = next((v for v in model_versions if v.current_stage == stage), None)
    return latest_version_info.run_id if latest_version_info else None

def load_model_and_vectorizer():
    model_name = "save_model"
    stage = "Production"
    run_id = get_latest_model_run_id(model_name, stage)
    if not run_id:
        raise Exception(f"No model found in the '{stage}' stage.")

    model_uri = f"runs:/{run_id}/model/model.pkl"
    model_path = mlflow.artifacts.download_artifacts(model_uri)
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")

    vectorizer_uri = f"runs:/{run_id}/vectorizer.pkl"
    vectorizer_path = mlflow.artifacts.download_artifacts(vectorizer_uri)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Vectorizer loaded successfully.")

    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments).tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
