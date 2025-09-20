# In api/index.py (Final Robust Version)
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_URL = os.environ.get("HF_API_URL")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

@app.route('/api/predict', methods=['POST'])
def predict():
    if not all([HF_API_URL, HF_API_TOKEN]):
        return jsonify({"error": "Server configuration is missing required environment variables."}), 500

    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "Request must include an 'image_url' in the JSON body."}), 400

    image_url = request.json['image_url']

    payload = {
        "data": [
            {"path": image_url, "meta": {"_type": "gradio.FileData"}}
        ]
    }

    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        hf_response = requests.post(HF_API_URL, headers=headers, json=payload)
        hf_response.raise_for_status()

        response_json = hf_response.json()

        # --- ROBUST CHECKING ---
        # First, check if Hugging Face returned an error message.
        if 'error' in response_json:
            return jsonify({"error": f"Hugging Face API Error: {response_json['error']}"}), 502

        # Next, check if the 'data' key is present as expected.
        if 'data' not in response_json or not isinstance(response_json['data'], list) or len(response_json['data']) == 0:
            return jsonify({
                "error": "Unexpected response format from Hugging Face API.",
                "details": response_json
            }), 502
        # --- END OF CHECKING ---

        # Extract the prediction data safely
        prediction_data = response_json["data"][0]
        top_prediction_label = prediction_data.get('label')
        confidence_score = 0.0

        if 'confidences' in prediction_data and isinstance(prediction_data['confidences'], list):
            for item in prediction_data['confidences']:
                if item.get('label') == top_prediction_label:
                    confidence_score = item.get('confidence', 0.0)
                    break

        return jsonify({
            "predicted_breed": top_prediction_label,
            "confidence_score": confidence_score
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Could not connect to Hugging Face API: {e}"}), 503
    except Exception as e:
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500