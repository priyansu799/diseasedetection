from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder='public')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_detection_model.h5')
model = tf.keras.models.load_model(model_path)

# Serve static HTML
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read and preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prob = model.predict(img)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    return jsonify({'label': label, 'prob': float(prob)})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
