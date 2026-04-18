from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import os

app = Flask(__name__)

# Load trained model
model = load_model("fabric_model.keras")

# Class labels
classes = ['cotton', 'silk', 'polyester', 'wool']

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    if file:
        # Save uploaded file
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        # Read & preprocess image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img = img.reshape(1, 100, 100, 3)

        # Predict
        pred = model.predict(img)
        class_index = np.argmax(pred)

        result = classes[class_index]
        confidence = np.max(pred) * 100

        return f"Fabric: {result} | Confidence: {confidence:.2f}%"

    return "No file uploaded"

if __name__ == "__main__":
    app.run(debug=True)