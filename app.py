from flask import Flask, request, render_template, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from werkzeug.utils import secure_filename
from functools import wraps
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
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
    MAX_CONTENT_LENGTH=30 * 1024 * 1024,
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
# Model loading
# ===============================
MODEL_NAME = "waleeyd/deepfake-detector-image"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    logging.info(f"Loading model: {MODEL_NAME}")
    
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    ).to(device)

    model.eval()
    torch.set_num_threads(8)

    logging.info(f"Model loaded successfully on {device}")

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

# ===============================
# Prediction logic (FIXED)
# ===============================
def predict_image(path):
    if not model or not processor:
        raise RuntimeError("Model not loaded")

    image = Image.open(path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_idx = logits.argmax(dim=-1).item()

    # 🔧 CRITICAL FIX: normalize id2label keys
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label = id2label[predicted_idx]

    probs = torch.softmax(logits, dim=-1)[0]
    confidence = float(probs[predicted_idx]) * 100

    final_label = "fake" if label.lower() in ["artificial", "ai", "fake"] else "real"

    return final_label, round(confidence, 2)

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
        image = Image.open(file).convert("RGB")
        image.save(path, "JPEG", quality=95)

        label, confidence = predict_image(path)
        file_hash = generate_file_hash(path)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "hash": file_hash
        })

    except Exception as e:
        logging.error(f"Predicti
