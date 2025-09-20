from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Get your Hugging Face details from Vercel's Environment Variables
HF_API_URL = os.environ.get("HF_API_URL")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

@app.route('/api/predict', methods=['POST'])
def predict():
    # --- Error Checking ---
    if not all([HF_API_URL, HF_API_TOKEN]):
        return jsonify({"error": "Server configuration missing"}), 500

    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "Request must include an 'image_url'"}), 400

    image_url = request.json['image_url']

    try:
        # 1. Download the image data from the URL provided by the frontend
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        image_data = image_response.content

        # 2. Call the Hugging Face API, passing the secret token in the headers
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        hf_response = requests.post(HF_API_URL, headers=headers, data=image_data)
        hf_response.raise_for_status()

        prediction_result = hf_response.json()

        # 3. Extract the top prediction and format it for our frontend
        top_prediction = prediction_result[0]

        return jsonify({
            "predicted_breed": top_prediction.get('label'),
            "confidence_score": top_prediction.get('score')
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching image or calling Hugging Face API: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500