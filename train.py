import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train_model():
    # Load extracted features
    df = pd.read_csv("features/features.csv")

    # Separate features and labels
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM model
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Save model + scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fault_detector.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 Model and scaler saved in models/ folder")

if __name__ == "__main__":
    train_model()
