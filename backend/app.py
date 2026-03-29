import os
import tempfile
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from service import classification_service

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# ─────────────────────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health_check():
    """Check API health and all loaded domains."""
    status = "healthy" if len(classification_service.domains) > 0 else "degraded"
    return jsonify({
        "status": status,
        "loaded_domains": list(classification_service.domains.keys()),
        "errors": classification_service.errors
    }), 200


# ─────────────────────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────────────────────
@app.route('/process', methods=['POST'])
def process_input():
    """
    Classify input text.

    Body (JSON):
        input   : str   – text to classify (required)
        domain  : str   – model domain to use, default 'base'

    Returns:
        message : predicted class label
        details : full prediction object (class, confidence, probabilities)
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        return jsonify({"error": "Invalid JSON"}), 400

    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    user_input = str(data.get('input', ''))
    domain     = data.get('domain', 'base')

    if not user_input:
        return jsonify({"error": "Field 'input' is required"}), 400

    try:
        result = classification_service.predict(user_input, domain)
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    if "error" in result:
        return jsonify(result), 500

    return jsonify({"message": result.get("class"), "details": result}), 200


# ─────────────────────────────────────────────────────────────
#  UPLOAD DATASET & TRAIN
# ─────────────────────────────────────────────────────────────
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """
    Upload a CSV and train a new domain model.

    Form fields:
        file        : CSV file (required)
        domain_name : str – name for the new domain (default 'custom')
        augmented   : str – 'true' to mark this as an augmented-data model
                           (stored in domain name as '<domain_name>_augmented')
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    domain_name = request.form.get('domain_name', 'custom')

    # Optional: caller can tag the domain as augmented
    is_augmented = request.form.get('augmented', 'false').lower() == 'true'
    if is_augmented:
        domain_name = f"{domain_name}_augmented"

    temp_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_path)

    result = classification_service.train_model(temp_path, domain_name)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result), 200


# ─────────────────────────────────────────────────────────────
#  LIST DOMAINS
# ─────────────────────────────────────────────────────────────
@app.route('/domains', methods=['GET'])
def list_domains():
    """Return all currently loaded model domains."""
    return jsonify({
        "domains": list(classification_service.domains.keys())
    }), 200


# ─────────────────────────────────────────────────────────────
#  LOAD DOMAIN ON DEMAND
# ─────────────────────────────────────────────────────────────
@app.route('/load_domain', methods=['POST'])
def load_domain():
    """
    Manually load a saved domain into memory.

    Body (JSON):
        domain : str – domain name to load (required)
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    domain = data.get('domain')
    if not domain:
        return jsonify({"error": "Field 'domain' is required"}), 400

    if domain in classification_service.domains:
        return jsonify({"message": f"Domain '{domain}' is already loaded"}), 200

    classification_service.load_resources(domain)

    if domain in classification_service.domains:
        return jsonify({"message": f"Domain '{domain}' loaded successfully"}), 200

    return jsonify({
        "error": f"Failed to load domain '{domain}'",
        "details": classification_service.errors
    }), 500


# ─────────────────────────────────────────────────────────────
#  COMPARE DOMAINS  (augmented vs normal, same input text)
# ─────────────────────────────────────────────────────────────
@app.route('/compare', methods=['POST'])
def compare_domains():
    """
    Run the same text through two domains and return both results side-by-side.
    Useful for comparing normal vs augmented model outputs.

    Body (JSON):
        input    : str  – text to classify (required)
        domain_a : str  – first domain  (default 'base')
        domain_b : str  – second domain (default 'base_augmented')
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    user_input = str(data.get('input', ''))
    if not user_input:
        return jsonify({"error": "Field 'input' is required"}), 400

    domain_a = data.get('domain_a', 'base')
    domain_b = data.get('domain_b', 'base_augmented')

    result_a = classification_service.predict(user_input, domain_a)
    result_b = classification_service.predict(user_input, domain_b)

    return jsonify({
        "input": user_input,
        domain_a: result_a,
        domain_b: result_b
    }), 200


if __name__ == '__main__':
    logger.info("Starting Flask server on port 5002...")
    app.run(debug=True, port=5002, host='0.0.0.0')