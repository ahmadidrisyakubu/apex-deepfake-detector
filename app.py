from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from werkzeug.utils import secure_filename
from functools import wraps
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
import numpy as np
import os
import time
import hashlib
import logging
import secrets

# ===============================
# App initialization
# ===============================
app = Flask(__name__)

app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    MAX_CONTENT_LENGTH=30 * 1024 * 1024,  # 30 MB
    UPLOAD_FOLDER="uploads"
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)
limiter.init_app(app)

# ===============================
# Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("security.log"), logging.StreamHandler()]
)

# ===============================
# Load Model from Hugging Face
# ===============================
MODEL_NAME = "waleeyd/deepfake-detector-image"

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    logging.info(f"Loading processor and model from {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()
    logging.info(f"Model loaded successfully on {device}")
    
    # Get labels from model config
    ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}
    logging.info(f"Model labels: {ID2LABEL}")
except Exception as e:
    logging.error(f"Model load failed: {e}")
    model = None
    processor = None

# ===============================
# Constants
# ===============================
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 30 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ===============================
# Helpers
# ===============================
def security_validate(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        logging.info(f"{request.remote_addr} → {request.endpoint}")
        return f(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    name, ext = os.path.splitext(secure_filename(filename))
    return f"{name}_{int(time.time())}{ext}"

def validate_file_security(file):
    if not file or file.filename == "":
        return ["No file provided"]

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    errors = []

    if size > MAX_FILE_SIZE:
        errors.append("File exceeds 30MB limit")

    if not allowed_file(file.filename):
        errors.append("Invalid file extension")

    try:
        img = Image.open(file)
        img.verify()
    except Exception:
        errors.append("Invalid or corrupted image file")
    finally:
        file.seek(0)

    return errors

def generate_file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def predict_image(path):
    """
    Predict if image is Real or Fake
    Logic: 
    - The model 'waleeyd/deepfake-detector-image' has labels like 'artificial' and 'real'.
    - We map 'artificial' (or any AI label) to 'fake' and 'real' to 'real'.
    """
    if not model or not processor:
        raise Exception("Model or processor not loaded")

    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    # Get predictions for all classes
    predictions = {ID2LABEL[i].lower(): float(probs[i]) for i in range(len(ID2LABEL))}
    
    # Logic to determine Fake vs Real
    fake_prob = 0.0
    real_prob = 0.0
    
    for label, prob in predictions.items():
        if "artificial" in label or "fake" in label:
            fake_prob += prob
        elif "real" in label:
            real_prob += prob
            
    if fake_prob > real_prob:
        label = "fake"
        confidence = fake_prob * 100
    else:
        label = "real"
        confidence = real_prob * 100
    
    return label, round(confidence, 2)

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET"])
@limiter.limit("30 per minute")
@security_validate
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@limiter.limit("60 per minute")
@security_validate
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    errors = validate_file_security(file)

    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    filename = sanitize_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        img = Image.open(file).convert("RGB")
        img.save(path, "JPEG", quality=95)

        label, confidence = predict_image(path)
        file_hash = generate_file_hash(path)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "hash": file_hash
        })

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500

    finally:
        if os.path.exists(path):
            os.remove(path)

# ===============================
# Security headers
# ===============================
@app.after_request
def security_headers(res):
    res.headers["X-Content-Type-Options"] = "nosniff"
    res.headers["X-Frame-Options"] = "DENY"
    res.headers["X-XSS-Protection"] = "1; mode=block"
    res.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    res.headers["Content-Security-Policy"] = (
        "default-src 'self' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
        "frame-src https://www.youtube.com https://www.youtube-nocookie.com;"
    )
    return res

# ===============================
# JSON Error Handlers
# ===============================
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large. Maximum allowed size is 30 MB."}), 413

@app.errorhandler(RateLimitExceeded)
def rate_limit_handler(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Invalid request or image file."}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error during prediction."}), 500

# ===============================
# Run
# ===============================
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
