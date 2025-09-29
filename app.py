import streamlit as st
import joblib
import numpy as np
import pandas as pd
from feature_extraction import extract_features

# Load model and scaler
model = joblib.load("models/fault_detector.pkl")
scaler = joblib.load("models/scaler.pkl")

# Class description mapping
class_descriptions = {
    "Bearing": "Defective bearing",
    "Flywheel": "Flywheel issue",
    "Healthy": "No defects (Healthy compressor)",
    "LIV": "Low inlet valve problem",
    "LOV": "Low outlet valve problem",
    "NRV": "Non-return valve problem",
    "Piston": "Piston issue",
    "Riderbelt": "Rider belt problem"
}

st.title("Air Compressor Fault Detection")
st.write("Upload one or more audio recordings of the compressor to predict possible defects.")

# Upload multiple files
uploaded_files = st.file_uploader("Choose WAV files", type="wav", accept_multiple_files=True)

if uploaded_files:
    results = []
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)  # Initialize progress bar

    with st.spinner("Processing audio files..."):
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            # Extract features
            features = extract_features(uploaded_file)
            features_scaled = scaler.transform([features])

            # Predict
            predicted_class = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]

            # Collect results
            results.append({
                "File": uploaded_file.name,
                "Prediction": predicted_class,
                "Description": class_descriptions[predicted_class],
                **{cls: f"{prob*100:.2f}%" for cls, prob in zip(model.classes_, probabilities)}
            })

            # Update progress bar
            progress_bar.progress(i / total_files)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # Display results
    st.subheader("Predictions for Uploaded Files")
    st.dataframe(df_results)

    # Download button
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='fault_predictions.csv',
        mime='text/csv'
    )
