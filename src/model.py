"""
Model training module.

This module trains a Random Forest classifier on the
breast cancer dataset and saves the trained model
and scaler for later inference.
"""
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from data_loader import load_data
from preprocessing import preprocess_features

# Ensure files are saved in project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    """
    Train a Random Forest model for cancer prediction.

    Loads data, preprocesses features, trains the model,
    evaluates accuracy, and saves the model and scaler.

    Returns:
        tuple: Trained model and fitted scaler.
    """
    dataframe = load_data()
    features = dataframe.drop("target", axis=1)
    target = dataframe["target"]

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    (
        features_train_scaled,
        features_test_scaled,
        scaler,
    ) = preprocess_features(features_train, features_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(features_train_scaled, target_train)

    # Save model and scaler in BASE_DIR
    joblib.dump(model, os.path.join(BASE_DIR, 'model.joblib'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.joblib'))

    accuracy = model.score(features_test_scaled, target_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    return model, scaler

if __name__ == "__main__":
    train_model()
