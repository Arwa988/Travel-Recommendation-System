import random
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import os
import time

# Initialize Flask application
app = Flask(__name__)

# Load and preprocess data
try:
    if not os.path.exists('travel_recommendation.csv'):
        raise FileNotFoundError("The file 'travel_recommendation.csv' was not found.")

    data = pd.read_csv('travel_recommendation.csv')

    # Check required columns
    required_columns = ['Name', 'Popularity', 'State', 'Type', 'BestTimeToVisit']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col} in the dataset.")

    data = data.dropna(subset=required_columns)  # Drop rows with missing values

    # Preprocessing: Round popularity values and categorize them
    data['popularity_rounded'] = data['Popularity'].apply(round)
    data['popularity_labels'] = [
        "Popular" if value == 9 else
        "A Bit Popular" if value == 8 else
        "Not Popular"
        for value in data['popularity_rounded']
    ]

    # Encode target variable
    label_encoder = LabelEncoder()
    data['Name_Encoded'] = label_encoder.fit_transform(data['Name'])

    # Feature encoding using pd.get_dummies
    encoded_features = pd.get_dummies(data[['State', 'Type', 'BestTimeToVisit', 'popularity_labels']], drop_first=True)
    x = encoded_features
    y = data['Name_Encoded']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Shuffle data to ensure randomness
    x_train, y_train = shuffle(x_train, y_train, random_state=None)

    # Initialize and train Logistic Regression Classifier
    clf_logistic = LogisticRegression(max_iter=1000, random_state=42)
    clf_logistic.fit(x_train, y_train)
    y_pred = clf_logistic.predict(x_test)

    print(f"Test set accuracy: {accuracy_score(y_test, y_pred)}")

except Exception as e:
    print(f"Error during data loading or model training: {e}")
    raise SystemExit

# Routes
@app.route('/')
def home():
    try:
        return render_template('index2.html')
    except Exception as e:
        return f"Error rendering template: {e}", 500


@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    try:
        random.seed(int(time.time()))  # Sets random seed based on current time
        user_input = request.json
        print(f"Received input: {user_input}")

        state = user_input.get('state')
        travel_type = user_input.get('type')
        best_time = user_input.get('time')
        popularity = user_input.get('popularity')

        if not all([state, travel_type, best_time, popularity]):
            return jsonify({'error': 'Missing input fields'}), 400

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'State': [state],
            'Type': [travel_type],
            'BestTimeToVisit': [best_time],
            'popularity_labels': [popularity]
        })
        print(f"Input data before encoding: {input_data}")

        # One-hot encode the input features
        input_encoded = pd.get_dummies(input_data, drop_first=True)

        # Ensure all training features are present, even if they weren't in the input
        input_encoded = input_encoded.reindex(columns=x.columns, fill_value=0)
        print(f"Reindexed encoded input: {input_encoded}")

        # Predict probabilities and select a random top-k result
        probabilities = clf_logistic.predict_proba(input_encoded)
        top_k_indices = np.argsort(probabilities[0])[-3:]  # Top 3 results
        random_index = random.choice(top_k_indices)
        recommendation = label_encoder.inverse_transform([random_index])[0]
        print(f"Prediction: {recommendation}, Top-K Probabilities: {probabilities}")

        # Post-prediction: Filter dataset to match user inputs for added variety
        filtered_data = data[
            (data['State'] == state) &
            (data['Type'] == travel_type) &
            (data['BestTimeToVisit'] == best_time) &
            (data['popularity_labels'] == popularity)
        ]

        if not filtered_data.empty:
            recommendation = random.choice(filtered_data['Name'].tolist())
            print("Random recommendation from filtered data.")

        return jsonify({'recommendation': recommendation})

    except Exception as e:
        print(f"Error in recommendation: {e}")
        return jsonify({'error': f"An error occurred: {e}"}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
