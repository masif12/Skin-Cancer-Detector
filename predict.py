import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model (update the path if necessary)
MODEL_PATH = "model/best_model.h5"
model = load_model(MODEL_PATH)

# Function to preprocess image and make prediction
def predict_image(image: Image.Image) -> str:
    # Preprocess the image (resize and normalize)
    image = image.resize((128, 128))  # Resize to the model input size
    image_array = np.array(image) / 255.0  # Normalize the pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(image_array)
    
    # Predict whether the lesion is benign or malignant
    if predictions[0][0] > 0.5:
        return "malignant"
    else:
        return "benign"
