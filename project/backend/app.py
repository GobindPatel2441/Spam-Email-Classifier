from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import predict_spam

app = Flask(__name__)
# Enable CORS for frontend integration
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No 'text' field provided"}), 400
        
        email_text = data["text"]
        result = predict_spam(email_text)
        
        # Result contains {"prediction": "Spam", "confidence": 0.92}
        return jsonify(result), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
