import os
import sys
import traceback
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# Disable GPU detection to save memory and startup time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress info/warning logs

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chess_cnn_model.h5")

# Global variable for lazy loading
_model = None

def load_model():
    """Load the model once and cache it globally."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        app.logger.warning(f"Model file '{MODEL_PATH}' not found. Using dummy model.")
        class DummyModel:
            def predict(self, x, verbose=0):
                return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _model = DummyModel()
        return _model

    try:
        _model = tf.keras.models.load_model(MODEL_PATH)
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        app.logger.error(traceback.format_exc())
        # Fallback to dummy model
        class DummyModel:
            def predict(self, x, verbose=0):
                return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _model = DummyModel()
    return _model

# Classes
CLASS_NAMES = [
    'bishop',
    'king',
    'knight',
    'pawn',
    'queen',
    'rook'
]

PIECE_EMOJIS = {
    'bishop': '♝',
    'king': '♚',
    'knight': '♞',
    'pawn': '♟',
    'queen': '♛',
    'rook': '♜'
}

def preprocess_image(file_storage):
    img = Image.open(file_storage.stream)
    img = img.convert("RGB")
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def perform_prediction(file_storage):
    model = load_model()  # Lazy load
    input_tensor = preprocess_image(file_storage)
    predictions = model.predict(input_tensor, verbose=0)

    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    class_name = CLASS_NAMES[predicted_index]
    emoji = PIECE_EMOJIS.get(class_name, '♟')

    return class_name, confidence, emoji

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        prediction=None,
        confidence=0.0,
        emoji='',
        error=None
    )

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return render_template(
            "index.html",
            prediction=None,
            confidence=0.0,
            emoji='',
            error="No image file provided."
        )

    file = request.files['image']
    if file.filename == '':
        return render_template(
            "index.html",
            prediction=None,
            confidence=0.0,
            emoji='',
            error="No image selected."
        )

    try:
        class_name, confidence, emoji = perform_prediction(file)
        app.logger.info(f"Prediction: {class_name} ({confidence:.2%})")
        return render_template(
            "index.html",
            prediction=class_name,
            confidence=confidence,
            emoji=emoji,
            error=None
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        app.logger.error(traceback.format_exc())
        return render_template(
            "index.html",
            prediction=None,
            confidence=0.0,
            emoji='',
            error=f"Prediction failed: {str(e)}"
        )

@app.route("/predict", methods=["GET"])
def predict_redirect():
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)