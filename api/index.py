# In api/index.py (Debug Version)
import requests
import json
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_API_URL = os.environ.get("HF_API_URL")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

@app.route('/api/predict', methods=['POST'])
def predict():
    # ... (The first part of the function is the same)
    if not all([HF_API_URL, HF_API_TOKEN]):
        return jsonify({"error": "Server configuration missing"}), 500
    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "Request must include 'image_url'"}), 400

    image_url = request.json['image_url']
    payload = {"data": [{"path": image_url, "meta": {"_type": "gradio.FileData"}}]}
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        initial_response = requests.post(HF_API_URL, headers=headers, json=payload)
        initial_response.raise_for_status()
        event_id = initial_response.json().get("event_id")

        if not event_id:
            raise ValueError("Could not get a valid event_id from the API.")

        stream_url = f"{HF_API_URL}/{event_id}"

        with requests.get(stream_url, headers=headers, stream=True) as stream_response:
            stream_response.raise_for_status()
            for line in stream_response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        try:
                            event_data = json.loads(decoded_line[5:])
                            if event_data.get("msg") == "process_completed":
                                # --- THIS IS THE CHANGE ---
                                # Instead of parsing, return the entire successful output payload.
                                # This will show us its exact structure.
                                return jsonify(event_data.get("output"))
                                # --- END OF CHANGE ---
                        except json.JSONDecodeError:
                            continue

        return jsonify({"error": "Stream ended without a final prediction."}), 504

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500