from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)

CORS(app)

# Load the model
model = load_model('maize_disease.h5')

class_labels = ['Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy']


# Image preprocessing function
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalization, adjust if necessary
    return image

# Endpoint to upload image and get prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust target size as needed

        prediction = model.predict(processed_image)
        # predicted_class = np.argmax(prediction, axis=-1)  # Modify based on your output
        predicted_class_idx = np.argmax(prediction, axis=-1)[0]
        predicted_class_name = class_labels[predicted_class_idx]
        
        return jsonify({
            'prediction_index': int(predicted_class_idx),
            'prediction_class': predicted_class_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
