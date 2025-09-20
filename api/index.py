from flask import Flask, request, jsonify
from gradio_client import Client
import os

app = Flask(__name__)

# Get the Space name and token from Environment Variables
# The gradio_client will automatically use the HF_API_TOKEN if it exists
HF_SPACE_NAME = os.environ.get("HF_SPACE_NAME")

# Initialize the client once when the app starts up
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
        # Use the client to make a blocking prediction call
        result = client.predict(
            image=image_url,
            api_name="/predict"
        )
        
        # The result from the client is the direct output from our Gradio app's function
        # The output is a dictionary of confidences, let's find the top one.
        top_breed = max(result, key=result.get)
        top_confidence = result[top_breed]

        return jsonify({
            "predicted_breed": top_breed,
            "confidence_score": top_confidence
        })

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500