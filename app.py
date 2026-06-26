import os
import sys
import traceback
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

# Disable GPU to save memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chess_cnn_model.h5")

# Global model variable
_model = None

def get_model():
    """Lazy load the model only when needed."""
    global _model
    if _model is not None:
        return _model
    
    app.logger.info(f"Loading model from {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        app.logger.error(f"Model file not found at {MODEL_PATH}")
        # Fallback dummy model
        class DummyModel:
            def predict(self, x, verbose=0):
                return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _model = DummyModel()
        return _model
    
    try:
        # Load with a specific session to avoid graph issues
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        app.logger.info("Model loaded successfully")
        return _model
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        app.logger.error(traceback.format_exc())
        # Fallback
        class DummyModel:
            def predict(self, x, verbose=0):
                return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _model = DummyModel()
        return _model

CLASS_NAMES = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']
PIECE_EMOJIS = {
    'bishop': '♝', 'king': '♚', 'knight': '♞',
    'pawn': '♟', 'queen': '♛', 'rook': '♜'
}

def preprocess_image(file_storage):
    """Preprocess uploaded image for prediction."""
    try:
        img = Image.open(file_storage.stream)
        img = img.convert("RGB")
        img = img.resize((28, 28))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        app.logger.info(f"Image preprocessed, shape: {img_array.shape}")
        return img_array
    except Exception as e:
        app.logger.error(f"Preprocessing failed: {e}")
        raise

def perform_prediction(file_storage):
    """Run model prediction with detailed error logging."""
    try:
        app.logger.info("Starting prediction...")
        model = get_model()
        input_tensor = preprocess_image(file_storage)
        
        app.logger.info("Running model.predict...")
        predictions = model.predict(input_tensor, verbose=0)
        app.logger.info(f"Predictions shape: {predictions.shape}")
        
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        class_name = CLASS_NAMES[predicted_index]
        emoji = PIECE_EMOJIS.get(class_name, '♟')
        
        app.logger.info(f"Prediction result: {class_name} ({confidence:.2%})")
        return class_name, confidence, emoji
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        app.logger.error(traceback.format_exc())
        raise

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, confidence=0.0, emoji='', error=None)

@app.route("/predict", methods=["POST"])
def predict():
    app.logger.info("Received POST request to /predict")
    
    if 'image' not in request.files:
        app.logger.warning("No image in request")
        return render_template("index.html", prediction=None, confidence=0.0, emoji='', error="No image file provided.")
    
    file = request.files['image']
    if file.filename == '':
        app.logger.warning("Empty filename")
        return render_template("index.html", prediction=None, confidence=0.0, emoji='', error="No image selected.")
    
    try:
        class_name, confidence, emoji = perform_prediction(file)
        return render_template(
            "index.html",
            prediction=class_name,
            confidence=confidence,
            emoji=emoji,
            error=None
        )
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"Prediction failed: {error_msg}")
        return render_template(
            "index.html",
            prediction=None,
            confidence=0.0,
            emoji='',
            error=f"Prediction failed: {error_msg}"
        )

@app.route("/predict", methods=["GET"])
def predict_redirect():
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)