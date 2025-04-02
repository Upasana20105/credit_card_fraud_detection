

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model_path = "fraud_detection_model.joblib"

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        input_data = data['data']
        input_array = np.array(input_data).reshape(1, -1) #Reshape for sklearn
        prediction = model.predict(input_array)[0] #extract prediction

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
