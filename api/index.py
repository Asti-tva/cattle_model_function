# In api/index.py

from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import io

app = Flask(__name__)

# --- MODEL LOADING ---
# We load the model outside of the request functions.
# This ensures the model is loaded only once when the server starts,
# making predictions much faster.

MODEL_PATH = 'cattle_breed_classifier_complete.pth' # IMPORTANT: Change this to your .pth file's name
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the entire model package
try:
    model_package = torch.load(MODEL_PATH, map_location=DEVICE)

    # Extract components
    CLASS_NAMES = model_package['class_names']
    model_config = model_package['model_config']
    
    # Initialize model from Hugging Face
    PROCESSOR = ViTImageProcessor.from_pretrained(model_config['model_name'])
    MODEL = ViTForImageClassification.from_pretrained(
        model_config['model_name'],
        num_labels=model_config['num_classes'],
        ignore_mismatched_sizes=True
    )
    
    # Load your fine-tuned weights
    MODEL.load_state_dict(model_package['model_state_dict'])
    MODEL.to(DEVICE)
    MODEL.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None

# --- API ENDPOINT ---

@app.route('/api/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model is not loaded"}), 500

    if not request.json or 'image_url' not in request.json:
        return jsonify({"error": "No image_url provided"}), 400

    image_url = request.json['image_url']

    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Open the image from the downloaded content
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Preprocess for ViT (adapted from your script)
        inputs = PROCESSOR(images=image, return_tensors="pt")
        processed_image = inputs['pixel_values'].to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = MODEL(processed_image)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_val = confidence.item()
        
        # Return the result as JSON
        return jsonify({
            "predicted_breed": predicted_class,
            "confidence_score": confidence_val
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True)