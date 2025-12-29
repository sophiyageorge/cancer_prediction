import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from data_loader import load_data
from preprocessing import preprocess_features

# Ensure files are saved in project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train_model():
    df = load_data()
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler in BASE_DIR
    joblib.dump(model, os.path.join(BASE_DIR, 'model.joblib'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler.joblib'))

    acc = model.score(X_test_scaled, y_test)
    print(f"Model Accuracy: {acc:.2f}")

    return model, scaler

if __name__ == "__main__":
    train_model()
