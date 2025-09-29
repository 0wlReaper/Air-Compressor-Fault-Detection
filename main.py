import numpy as np
import pandas as pd
import joblib
from feature_extraction import extract_features
model = joblib.load("models/fault_detector.pkl")
scaler = joblib.load("models/scaler.pkl")
# Ask user for file path or hardcode
file_path = input("Enter the path of the audio file: ")

try:
    # Extract features
    features = extract_features(file_path)

    # Scale features
    features_scaled = scaler.transform([features])  # wrap in a list to make 2D

    # Predict class
    prediction = model.predict(features_scaled)
    print("Predicted class:", prediction[0])

    # Optional: show class probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_scaled)
        print("Class probabilities:", proba)

except Exception as e:
    print(f"Error processing file: {e}")
