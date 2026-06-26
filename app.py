import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

MODEL_PATH = 'chess_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']
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
    img = img.convert('RGB')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def perform_prediction(file):
    input_tensor = preprocess_image(file)
    predictions = model.predict(input_tensor, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    class_name = CLASS_NAMES[predicted_index]
    emoji = PIECE_EMOJIS.get(class_name, '♟')
    return class_name, confidence, emoji

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
                           prediction=None,
                           confidence=0.0,
                           emoji='',
                           error=None)

@app.route('/', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html',
                               prediction=None,
                               confidence=0.0,
                               emoji='',
                               error='No image file provided.')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html',
                               prediction=None,
                               confidence=0.0,
                               emoji='',
                               error='No image selected.')
    try:
        class_name, confidence, emoji = perform_prediction(file)
        print(f"Prediction: {class_name} ({confidence:.2%})")  # debug
        return render_template('index.html',
                               prediction=class_name,
                               confidence=confidence,
                               emoji=emoji,
                               error=None)
    except Exception as e:
        return render_template('index.html',
                               prediction=None,
                               confidence=0.0,
                               emoji='',
                               error=f'Prediction failed: {str(e)}')

@app.route('/predict', methods=['GET'])
def predict_get():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)