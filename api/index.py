# In api/index.py (Final Version with Streaming Logic)
import requests
import json
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_API_URL = os.environ.get("HF_API_URL")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

@app.route('/api/predict', methods=['POST'])
def predict():
    if not all([HF_API_URL, HF_API_TOKEN]):
        return jsonify({"error": "Server configuration missing"}), 500

    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "Request must include 'image_url'"}), 400

    image_url = request.json['image_url']
    payload = {"data": [{"path": image_url, "meta": {"_type": "gradio.FileData"}}]}
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    try:
        # STEP 1: Post the job and get the event_id
        initial_response = requests.post(HF_API_URL, headers=headers, json=payload)
        initial_response.raise_for_status()

        initial_data = initial_response.json()
        event_id = initial_data.get("event_id")

        if not event_id:
            # Handle cases where Gradio might respond directly for very fast jobs
            if "data" in initial_data:
                prediction_data = initial_data["data"][0]
                # (Simplified parsing for direct response)
                return jsonify({"predicted_breed": prediction_data.get('label')})
            else:
                raise ValueError("API did not return an event_id or a direct result.")

        # STEP 2: Use the event_id to stream for the result
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
                                # This is the final, successful prediction
                                prediction_data = event_data["output"]["data"][0]
                                top_prediction = prediction_data.get('label')
                                confidence = 0.0
                                if 'confidences' in prediction_data and prediction_data['confidences']:
                                    for c in prediction_data['confidences']:
                                        if c.get('label') == top_prediction:
                                            confidence = c.get('confidence', 0)
                                            break

                                return jsonify({
                                    "predicted_breed": top_prediction,
                                    "confidence_score": confidence
                                })
                        except json.JSONDecodeError:
                            continue # Ignore non-JSON lines

        return jsonify({"error": "Prediction stream ended without a result."}), 504

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500