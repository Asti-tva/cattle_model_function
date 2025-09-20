# In api/index.py (Corrected Version)
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_URL = os.environ.get("HF_API_URL")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

@app.route('/api/predict', methods=['POST'])
def predict():
    if not all([HF_API_URL, HF_API_TOKEN]):
        return jsonify({"error": "Server configuration missing"}), 500

    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "Request must include an 'image_url'"}), 400

    image_url = request.json['image_url']

    # --- THIS IS THE CORRECTED PART ---
    # We must build the exact JSON payload the Gradio API expects.
    payload = {
      "data": [
        {
          "path": image_url,
          "meta": {"_type": "gradio.FileData"}
        }
      ]
    }
    # --- END OF CORRECTION ---

    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

        # Send the payload as JSON
        hf_response = requests.post(HF_API_URL, headers=headers, json=payload)
        hf_response.raise_for_status()

        # The result is nested inside a 'data' list
        prediction_data = hf_response.json()["data"][0]

        return jsonify({
            "predicted_breed": prediction_data['label'],
            "confidence_score": prediction_data['confidences'][0]['confidence']
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error calling Hugging Face API: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500