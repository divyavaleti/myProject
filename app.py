from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import json

app = Flask(__name__)

# Load the model from JSON and HDF5 files
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route to handle the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    
    # Save the file
    file.save('static/uploaded_image.jpg')
    
    # Preprocess the image
    img_array = preprocess_image('static/uploaded_image.jpg')
    
    # Make prediction
    prediction = loaded_model.predict(img_array)
    
    # Get the predicted class (assuming binary classification)
    predicted_class = "Real" if prediction[0][0] > 0.5 else "Fake"
    
    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)