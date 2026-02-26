"""
PhishGuard ML-Enhanced Analyzer
Trained ML model use kar ke predictions deta hai

Ye file trained model ko load karti hai aur
URLs ko analyze kar ke ML-based predictions deti hai


"""

import os
import re
from urllib.parse import urlparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline




# ============================================
# FEATURE EXTRACTION (Same as training)
# ============================================

def extract_features_for_ml(url):
    """
    URL se ML ke liye features extract karta hai
    Training me use hue same features

    Parameters:
        url (string): Analyze karne wala URL

    Returns:
        list: 20 numerical features
    """

    parsed = urlparse(url)
    features = []

    # 1. URL Length
    features.append(len(url))

    # 2. Domain Length
    features.append(len(parsed.netloc))

    # 3. Dot Count
    features.append(url.count('.'))

    # 4. Dash Count
    features.append(url.count('-'))

    # 5. Underscore Count
    features.append(url.count('_'))

    # 6. Slash Count
    features.append(url.count('/'))

    # 7. Question Mark
    features.append(url.count('?'))

    # 8. Equal Sign
    features.append(url.count('='))

    # 9. At Symbol
    features.append(1 if '@' in url else 0)

    # 10. Ampersand
    features.append(url.count('&'))

    # 11. Double Slash Count
    features.append(url.count('//'))

    # 12. Digit Count
    features.append(sum(c.isdigit() for c in url))

    # 13. Letter Count
    features.append(sum(c.isalpha() for c in url))

    # 14. Has IP Address
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    features.append(1 if re.search(ip_pattern, url) else 0)

    # 15. HTTPS Check
    features.append(1 if parsed.scheme == 'https' else 0)

    # 16. Suspicious TLD
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.pw', '.top']
    features.append(1 if any(url.endswith(tld) for tld in suspicious_tlds) else 0)

    # 17. Suspicious Keywords Count
    suspicious_keywords = ['login', 'secure', 'account', 'verify', 'paypal',
                           'bank', 'update', 'confirm', 'suspend', 'locked',
                           'password', 'payment', 'billing']
    features.append(sum(1 for word in suspicious_keywords if word in url.lower()))

    # 18. Has "www"
    features.append(1 if 'www' in url.lower() else 0)

    # 19. URL Shortener
    shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly']
    features.append(1 if any(short in parsed.netloc for short in shorteners) else 0)

    # 20. Subdomain Count
    subdomain_count = parsed.netloc.count('.')
    features.append(subdomain_count)

    return features


# ============================================
# MODEL LOADING
# ============================================

# Global variable for model
_ml_model = None


def load_ml_model(model_path='phishing_model.pkl'):
    """
    Trained ML model ko load karta hai

    Ye function model ko memory me load karta hai
    Sirf ek baar load hoga (first call pe)

    Parameters:
        model_path: Model file ka path

    Returns:
        model: Loaded model ya None (agar file nahi mili)
    """

    global _ml_model

    # Agar model already loaded hai toh return kar do
    if _ml_model is not None:
        return _ml_model

    # Model file exist karti hai ya nahi check karo
    if not os.path.exists(model_path):
        print(f"⚠️  Warning: ML model not found at {model_path}")
        print(f"💡 Run 'train_model.py' first to create the model")
        return None

    try:
        # Model load karo
        with open(model_path, 'rb') as f:
            _ml_model = pickle.load(f)

        print(f"✅ ML model loaded successfully from {model_path}")
        return _ml_model

    except Exception as e:
        print(f"❌ Error loading ML model: {e}")
        return None


# ============================================
# ML PREDICTION
# ============================================

def predict_with_ml(url, model_path='phishing_model.pkl'):
    """
    ML model se URL ko predict karta hai

    Parameters:
        url (string): Analyze karne wala URL
        model_path: Model file path

    Returns:
        dict: Prediction results with confidence
    """

    # Model load karo (agar already loaded nahi hai)
    model = load_ml_model(model_path)

    if model is None:
        # Agar model nahi hai toh None return karo
        return None

    try:
        # Model expects RAW URL (not engineered features)
        # Our pipeline: TfidfVectorizer → RandomForest handles everything

        prediction = model.predict([url])[0]  # URL → TF-IDF → prediction
        probabilities = model.predict_proba([url])[0]  # Same URL for both!

        is_phishing = (prediction == 1)
        confidence = probabilities[1] if is_phishing else probabilities[0]

        result = {
            'ml_prediction': 'phishing' if is_phishing else 'legitimate',
            'ml_confidence': round(confidence * 100, 2),
            'ml_prob_phishing': round(probabilities[1] * 100, 2),
            'ml_prob_legitimate': round(probabilities[0] * 100, 2),
        }

        return result

    except Exception as e:
        print(f"❌ ML prediction error: {e}")
        return None


# ============================================
# ENHANCED ANALYZE FUNCTION
# ============================================



def analyze_url_with_ml(url):
    parsed = urlparse(url)
    parsed_domain = parsed.netloc.lower().lstrip("www.")


    # ================= TRUSTED DOMAINS =================
    TRUSTED_DOMAINS = [
    "google.com",
    "www.google.com",
    "facebook.com",
    "www.facebook.com",
    "youtube.com",
    "www.youtube.com",
    "github.com",
    "www.github.com"
]

    if any(parsed_domain == d or parsed_domain.endswith("." + d) for d in TRUSTED_DOMAINS):
        return {
            "risk_score": 0,
            "warnings": [],
            "ml_prediction": "legitimate",
            "ml_confidence": 99.9,
            "ml_prob_phishing": 0.1,
            "ml_prob_legitimate": 99.9,
            "final_verdict": "✅ SAFE (Trusted Domain)"
        }

    
    # ================= RULE-BASED ANALYSIS =================
    risk_score = 0
    warnings_list = []

    if len(url) > 75:
        risk_score += 15
        warnings_list.append("Very long URL detected")
    elif len(url) > 54:
        risk_score += 10
        warnings_list.append("Long URL")

    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    if re.search(ip_pattern, url):
        risk_score += 40
        warnings_list.append("IP address used instead of domain")

    if parsed.scheme != 'https':
        warnings_list.append("Connection is not HTTPS")

    suspicious_keywords = [
        'login','secure','account','verify','paypal',
        'bank','update','confirm','suspend'
    ]
    keyword_count = sum(1 for w in suspicious_keywords if w in url.lower())
    if keyword_count:
        risk_score += keyword_count * 8
        warnings_list.append(f"{keyword_count} suspicious keywords found")

    risk_score = min(risk_score, 100)

    result = {
        "risk_score": risk_score,
        "warnings": warnings_list
    }

    # ================= ML PREDICTION =================
    ml_result = predict_with_ml(url)

    if not ml_result:
        result["final_verdict"] = "⚠️ ML model unavailable"
        return result

    result.update(ml_result)
    phish_prob = ml_result["ml_prob_phishing"]

    # 🔒 SAFETY CLAMP
    if risk_score < 15 and phish_prob > 60:
        phish_prob = 20
        result["ml_prob_phishing"] = 20
        result["ml_confidence"] = 80

    # ================= FINAL VERDICT =================
    if (
    phish_prob >= 85 and
    risk_score >= 70 and
    re.search(ip_pattern, url)
):
        final_verdict = "🚨 CONFIRMED PHISHING"
    elif phish_prob >= 65:
        final_verdict = "⚠️ PHISHING"
    elif phish_prob >= 45:
        final_verdict = "🟠 SUSPICIOUS"
    else:
        final_verdict = "✅ SAFE"

    result["final_verdict"] = final_verdict
    return result


# ============================================
# BACKWARD COMPATIBILITY
# ============================================

def analyze_url(url):
    """
    Original function name - backward compatibility ke liye
    Flask app me ye hi function call hoga
    """
    return analyze_url_with_ml(url)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("🛡️ PHISHGUARD ML-ENHANCED ANALYZER")
    print("=" * 60)

    # 🔥 USER INPUT MODE
    while True:
        url = input("\n🔗 Enter URL to test (or 'quit' to exit): ").strip()

        if url.lower() in ['quit', 'exit', 'q']:
            print("👋 Thanks for using PHISGUARD!")
            break

        if not url:
            print("⚠️ Please enter a valid URL!")
            continue

        print(f"\n{'=' * 60}")
        print(f"🔍 Testing: {url}")
        print(f"{'=' * 60}")

        # Tumhara main analysis function call karo
        result = analyze_url(url)  # ya jo function name hai

        if result:
            print(f"\n🎯 FINAL VERDICT: {'🚨 PHISHING' if result['ml_prediction'] == 'phishing' else '✅ SAFE'}")
            print(f"   Confidence: {result['ml_confidence']}%")
        else:
            print("❌ Analysis failed!")

        print("-" * 60)

"""
============================================
USAGE INSTRUCTIONS
============================================

1. First, train the model:
   python train_model.py

2. This creates: phishing_model.pkl

3. Then use this analyzer:
   from analyzer import analyze_url
   result = analyze_url("http://suspicious-url.com")

4. Result contains:
   - risk_score: Rule-based score (0-100)
   - ml_prediction: ML classification
   - ml_confidence: ML confidence percentage
   - final_verdict: Combined decision
   - warnings: List of detected issues

5. If model is not found:
   - Falls back to rule-based only
   - Warns in console
   - Still provides analysis

============================================
INTEGRATION WITH FLASK
============================================

In app.py, no changes needed!
Just replace analyzer.py with this file.

The analyze_url() function works exactly the same,
but now includes ML predictions automatically.

============================================
"""