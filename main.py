from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)

# Load model
load_dotenv()

LATEST_MODEL_URL = os.getenv("LATEST_MODEL_URL")
DESTINATION_MODEL_PATH = os.getenv("DESTINATION_MODEL_PATH")

response = requests.get(LATEST_MODEL_URL)

with open(DESTINATION_MODEL_PATH, 'wb') as f:
    f.write(response.content)

model = load_model(DESTINATION_MODEL_PATH)

class_names = ['calculus', 'caries', 'gingivitis', 'hypodontia', 'tooth_discoloration', 'ulcer']

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    img = tf.io.decode_image(file.read(), channels=3)
    img = tf.image.resize(img, (224, 224))  # Sesuaikan ukuran input model
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Check confidence level
    if confidence < 0.9:
        return jsonify({"class": "Not detected", "confidence": float(confidence)})

    # Return JSON response
    return jsonify({"class": predicted_class, "confidence": float(confidence)})

# Run server: python filename.py
if __name__ == "__main__":
    app.run(debug=True)