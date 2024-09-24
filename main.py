from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model architecture from JSON
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load the model weights
model.load_weights('model.h5')

# Rest of the code remains the same...
def preprocess_image(img_path, target_size=(256, 256)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]
    return img_array

# Function to make a prediction
def predict_logo(img_path):
    processed_image = preprocess_image(img_path)
    prediction = model.predict(processed_image)

    if prediction[0][0] > 0.5:
        return "Real Logo"
    else:
        return "Fake Logo"

# Get user input for image file path
image_path = "4.jpg"

# Perform prediction
result = predict_logo(image_path)
print("Prediction:", result)