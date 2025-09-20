# In api/index.py (Final Corrected Version)
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file # <-- Import handle_file
import os

app = Flask(__name__)

HF_SPACE_NAME = os.environ.get("HF_SPACE_NAME")

try:
    if not HF_SPACE_NAME:
        raise ValueError("HF_SPACE_NAME environment variable is not set.")
    client = Client(HF_SPACE_NAME)
    print("Successfully initialized Gradio client.")
except Exception as e:
    client = None
    print(f"Error initializing Gradio client: {e}")

@app.route('/api/predict', methods=['POST'])
def predict():
    if client is None:
        return jsonify({"error": "Gradio client is not available."}), 500

    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "Request must include 'image_url'."}), 400

    image_url = request.json['image_url']

    try:
        # --- THIS IS THE FINAL FIX ---
        # Use the handle_file function to correctly format the URL
        result = client.predict(
            image=handle_file(image_url), # <-- This is the change
            api_name="/predict"
        )
        # --- END OF FIX ---

        # The result is the dictionary from our Gradio app's function. Find the top prediction.
        top_breed = max(result, key=result.get)
        top_confidence = result[top_breed]

        return jsonify({
            "predicted_breed": top_breed,
            "confidence_score": top_confidence
        })

    except Exception as e:
        # The client library can raise specific errors, let's return a clean message
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500