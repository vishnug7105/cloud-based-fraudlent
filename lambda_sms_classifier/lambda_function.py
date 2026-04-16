import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import json
import joblib
from pathlib import Path
from typing import Any, Dict
from joblib import parallel_backend

# Global cache (persists across warm Lambda invocations)
model = None

def load_model():
    """
    Load the trained model from disk only once.
    """
    global model
    if model is None:
        model_path = Path(__file__).parent / "sms_spam_model.joblib"
        model = joblib.load(model_path)
    return model

def safe_predict_proba(model, X):
    """
    Predict probabilities using joblib in a Lambda-safe way.
    Falls back to serial execution if parallel processing fails.
    """
    try:
        # Use loky backend with /tmp for safe temp folder in Lambda
        with parallel_backend('loky', temp_folder='/tmp'):
            return model.predict_proba(X)
    except PermissionError:
        # Fallback to serial execution
        return model.predict_proba(X)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for SMS spam detection.

    Expected input (API Gateway):
    {
        "body": "{\"text\": \"Your message here\"}"
    }
    """
    try:
        # Parse request body
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event["body"])
        else:
            body = event

        # Extract text
        text = body.get("text")

        # Validate input
        if not isinstance(text, str) or not text.strip():
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Invalid or missing 'text' field"})
            }

        # Load model
        clf = load_model()

        # Predict probabilities safely
        proba = safe_predict_proba(clf, [text])[0]
        p_ham, p_spam = float(proba[0]), float(proba[1])

        # Assign label
        label = "spam" if p_spam >= 0.5 else "ham"

        # Logging (visible in CloudWatch)
        print(f"Input: {text[:50]}")
        print(f"Prediction: {label}, Spam probability: {p_spam}")

        # Build response
        response_body = {
            "label": label,
            "probabilities": {
                "ham": p_ham,
                "spam": p_spam
            }
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(response_body)
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }