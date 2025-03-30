# credit_card_fraud_detection.py (Machine Learning Model)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_and_save_model(data_path, model_path):
    """Trains a Random Forest model and saves it."""
    try:
        df = pd.read_csv(data_path)
        X = df.drop('Class', axis=1)  # 'Class' is the target variable (0: legitimate, 1: fraud)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, model_path)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        return True  # Indicate success

    except Exception as e:
        print(f"Error training model: {e}")
        return False
def load_and_predict(model_path, input_data):
      """Loads a trained model and makes predictions."""
      try:
          model = joblib.load(model_path)
          prediction = model.predict([input_data]) #input_data must be a list of lists.
          return prediction[0]
      except Exception as e:
          print(f"Error loading model or predicting: {e}")
          return None

if __name__ == "__main__":
    data_path = "creditcard.csv"  # Replace with your dataset path
    model_path = "fraud_detection_model.joblib"

    if train_and_save_model(data_path, model_path):
        print("Model trained and saved successfully.")
    else:
        print("Model training failed.")