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

class_labels = ['Gray_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight', 'Healthy']


# Image preprocessing function
def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0 
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image, target_size=(224, 224)) 

        prediction = model.predict(processed_image)
        # predicted_class = np.argmax(prediction, axis=-1)
        predicted_class_idx = np.argmax(prediction, axis=-1)[0]
        predicted_class_name = class_labels[predicted_class_idx]
        predicted_class_confidence = float(np.max(prediction))
        
        return jsonify({
            'prediction_index': int(predicted_class_idx),
            'prediction_class': predicted_class_name,
            'confidence': predicted_class_confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
