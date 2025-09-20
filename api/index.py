# In api/index.py (Returning Top 3)
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
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
        result = client.predict(
            image=handle_file(image_url),
            api_name="/predict"
        )

        # --- THIS IS THE UPDATED PART ---
        # The 'result' dictionary contains a 'confidences' list with the top 3 predictions.
        # We will reformat it into a clean list for our API response.

        predictions_list = result.get('confidences', [])

        formatted_predictions = [
            {"breed": item.get('label'), "score": item.get('confidence')}
            for item in predictions_list
        ]
        # --- END OF UPDATE ---

        return jsonify({"predictions": formatted_predictions})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500