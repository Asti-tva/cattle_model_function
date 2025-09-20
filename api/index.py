# In api/index.py (Proof-of-Life Test)
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    # This is a test to prove the new code is live.
    # It does not call the Hugging Face API.
    return jsonify({
        "message": "The new deployment is working!",
        "version": "proof_of_life_v2" 
    })