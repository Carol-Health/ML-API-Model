# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import numpy as np
# import os
# import requests
# from dotenv import load_dotenv

# app = Flask(__name__)

# # Load model
# load_dotenv()

# LATEST_MODEL_URL = os.getenv("LATEST_MODEL_URL")
# DESTINATION_MODEL_PATH = os.getenv("DESTINATION_MODEL_PATH")

# response = requests.get(LATEST_MODEL_URL)

# with open(DESTINATION_MODEL_PATH, 'wb') as f:
#     f.write(response.content)

# model = load_model(DESTINATION_MODEL_PATH)

# class_names = ['calculus', 'caries', 'gingivitis', 'hypodontia', 'tooth_discoloration', 'ulcer']

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files['file']
#     img = tf.io.decode_image(file.read(), channels=3)
#     img = tf.image.resize(img, (224, 224))  # Sesuaikan ukuran input model
#     img = img / 255.0  # Normalisasi
#     img = np.expand_dims(img, axis=0)

#     # Predict
#     predictions = model.predict(img)
#     predicted_class = class_names[np.argmax(predictions)]
#     confidence = np.max(predictions)

#     # Check confidence level
#     if confidence < 0.9:
#         return jsonify({"class": "Not detected", "confidence": float(confidence)})

#     # Return JSON response
#     return jsonify({"class": predicted_class, "confidence": float(confidence)})

# # Run server: python filename.py
# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import requests
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore

app = Flask(__name__)

# Load environment variables
load_dotenv()

# # Firebase Admin SDK initialization
# FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
# if not FIREBASE_CREDENTIALS_PATH or not os.path.exists(FIREBASE_CREDENTIALS_PATH):
#     raise FileNotFoundError("Firebase credentials file not found.")

# Initialize Firebase Admin SDK with service account
# cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app()

# Initialize Firestore client
db = firestore.client()

# Load model
LATEST_MODEL_URL = os.getenv("LATEST_MODEL_URL")
DESTINATION_MODEL_PATH = os.getenv("DESTINATION_MODEL_PATH")

# Download the model from the specified URL
response = requests.get(LATEST_MODEL_URL)
with open(DESTINATION_MODEL_PATH, 'wb') as f:
    f.write(response.content)

# Load the model
model = load_model(DESTINATION_MODEL_PATH)

class_names = ['calculus', 'caries', 'gingivitis', 'hypodontia', 'tooth_discoloration', 'ulcer']

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    img = tf.io.decode_image(file.read(), channels=3)
    img = tf.image.resize(img, (224, 224))  # Resize sesuai model input
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Jika confidence rendah
    if confidence < 0.9:
        return jsonify({
            "class": "Not detected",
            "confidence": float(confidence),
            "message": "Confidence is too low for reliable prediction."
        })

    # Ambil data dari Firestore
    disease_data = get_disease_info(predicted_class)

    return jsonify({
        "class": predicted_class,
        "confidence": float(confidence),
        "description": disease_data.get("description", "No description available"),
        "treatment": disease_data.get("treatment", "No treatment available")
    })

def get_disease_info(diseases_class):
    """
    Mengambil deskripsi dan treatment dari Firestore berdasarkan kelas penyakit.
    """
    try:
        doc_ref = db.collection("diseases").document(diseases_class)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            return {"description": "No description found", "treatment": "No treatment found"}
    except Exception as e:
        return {"description": f"Error retrieving data: {str(e)}", "treatment": ""}

# Run server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
