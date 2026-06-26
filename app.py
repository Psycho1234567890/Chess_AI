import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

CLASS_NAMES = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']
PIECE_EMOJIS = {
    'bishop': '♝', 'king': '♚', 'knight': '♞',
    'pawn': '♟', 'queen': '♛', 'rook': '♜'
}

# Dummy model - always returns pawn
class DummyModel:
    def predict(self, x, verbose=0):
        return np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])

model = DummyModel()

def preprocess_image(file_storage):
    img = Image.open(file_storage.stream)
    img = img.convert("RGB")
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def perform_prediction(file_storage):
    input_tensor = preprocess_image(file_storage)
    predictions = model.predict(input_tensor, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    class_name = CLASS_NAMES[predicted_index]
    emoji = PIECE_EMOJIS.get(class_name, '♟')
    return class_name, confidence, emoji

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", prediction=None, confidence=0.0, emoji='', error=None)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", prediction=None, confidence=0.0, emoji='', error="No image file provided.")
    file = request.files['image']
    if file.filename == '':
        return render_template("index.html", prediction=None, confidence=0.0, emoji='', error="No image selected.")
    try:
        class_name, confidence, emoji = perform_prediction(file)
        return render_template("index.html", prediction=class_name, confidence=confidence, emoji=emoji, error=None)
    except Exception as e:
        return render_template("index.html", prediction=None, confidence=0.0, emoji='', error=f"Prediction failed: {str(e)}")

@app.route("/predict", methods=["GET"])
def predict_redirect():
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)