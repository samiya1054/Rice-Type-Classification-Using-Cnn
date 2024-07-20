import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import warnings
import cv2
from flask import Flask, request, render_template
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model once when the server starts
custom_objects = {'KerasLayer': hub.KerasLayer}
model_path = 'rice_classify.h5'

try:
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            f = request.files['image']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            f.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.resize(img, (224, 224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            pred = pred.argmax()

            df_labels = {
                0: 'arborio',
                1: 'basmati',
                2: 'ipsala',
                3: 'jasmine',
                4: 'karacadag'
            }

            prediction = df_labels.get(pred, "Unknown")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            prediction = "Error during prediction"

        return render_template('result.html', prediction_text=prediction)
    return render_template('result.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)



