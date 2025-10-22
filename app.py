from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)  # ✅ Allow cross-origin requests from your React frontend

# Load model path
MODEL_DIR = os.environ.get("URDU_PHISH_MODEL_PATH", "./model")
nlp = None

# Load the model lazily
def get_pipeline():
    global nlp
    if nlp is None:
        print("🔁 Loading Urdu PhishGuard model from:", MODEL_DIR)
        nlp = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
    return nlp


# ✅ Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Urdu PhishGuard AI backend running"}), 200


# ✅ Main analysis route for React
@app.route("/api/analyze", methods=["POST"])
def analyze_message():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    message = data["message"].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    pipe = get_pipeline()
    result = pipe(message)[0]

    # Apply thresholding logic
    score = float(result["score"])
    raw_label = result["label"]

    if raw_label == "LABEL_1" and score >= 0.70:
        label = "PHISHING"
    elif raw_label == "LABEL_1" and score < 0.70:
        label = "SUSPICIOUS"
    else:
        label = "SAFE"

    response = {
        "classification": label,
        "confidence": round(score * 100, 2),
        "details": (
            "This message shows strong phishing indicators." if label == "PHISHING"
            else "This message may require human review." if label == "SUSPICIOUS"
            else "No phishing patterns detected. Appears safe."
        )
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
