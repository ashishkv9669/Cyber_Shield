from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from analyzer import analyze_url
from deepfake_detector import analyze_deepfake
import traceback
import requests


app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ------------------------
# Home route
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------
# Scan URL API (UNCHANGED)
# ------------------------
@app.route("/scan-url", methods=["POST"])
def scan_url():
    try:
        data = request.get_json(force=True)
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "URL is required"}), 400

        result = analyze_url(url)

        if not result:
            return jsonify({"error": "Analysis failed"}), 500

        return jsonify({
            "url": url,
            "ml_prediction": result.get("ml_prediction"),
            "ml_confidence": result.get("ml_confidence"),
            "ml_prob_phishing": result.get("ml_prob_phishing"),
            "ml_prob_legitimate": result.get("ml_prob_legitimate"),
            "final_verdict": result.get("final_verdict"),
            "risk_score": result.get("risk_score"),
            "warnings": result.get("warnings", [])
        })

    except Exception:
        print("❌ Backend Error:")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# =========================================================
# 📧 EMAIL BREACH CHECK API
# =========================================================
@app.route("/scan-email", methods=["POST"])
def scan_email():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()

        if not email:
            return jsonify({"final_verdict": "No email provided"}), 400

        url = f"https://leakcheck.io/api/public?check={email}"
        response = requests.get(url)
        result = response.json()

        if result.get("success") and result.get("found", 0) > 0:

            breach_count = result["found"]

            if breach_count >= 5:
                verdict = f"HIGH RISK ❌ Found in {breach_count} breaches"
            else:
                verdict = f"WARNING ⚠️ Found in {breach_count} breach"

        else:
            verdict = "SAFE ✅ No breach found"

        return jsonify({"final_verdict": verdict})

    except Exception:
        print("❌ EMAIL SCAN ERROR:")
        traceback.print_exc()
        return jsonify({"final_verdict": "Error checking email"}), 500


# =========================================================
# 🎭 DEEPFAKE SCAN API
# =========================================================
@app.route("/scan-deepfake", methods=["POST"])
def scan_deepfake():

    try:
        # 🔍 debug print
        print("FILES:", request.files)

        if "file" not in request.files:
            return jsonify({"error": "No file received"}), 200

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 200

        result = analyze_deepfake(file)

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 200
    
        


# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    print("🌐 Starting PHISHGUARD...")
    app.run(host="0.0.0.0", port=8080, debug=True)