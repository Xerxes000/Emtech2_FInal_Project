import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import gdown
import os
import h5py

# Google Drive File ID for the model
GOOGLE_DRIVE_URL = "https://drive.google.com/file/d/1bKAnp_btGozeaQqJbfHkE6-AePZOcYT7/view?usp=sharing"
MODEL_PATH = "mobilenet_cifar10_model.h5"

# Function to validate HDF5 file format
def is_valid_h5_file(file_path):
    try:
        with h5py.File(file_path, "r") as f:
            return True
    except Exception as e:
        st.error(f"Invalid HDF5 file: {e}")
        return False

# Function to download the model if not already downloaded
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model... This may take a while.")
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)

# Download and validate the model file
download_model()
if not is_valid_h5_file(MODEL_PATH):
    st.error("Model file is corrupted or not a valid HDF5 file.")
    st.stop()

# Load the model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# CIFAR-10 class names
classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

# Streamlit app
st.title("Image Classification with MobileNet")
st.write("Upload an image, and the model will classify it into one of the CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Process the uploaded image
        image = Image.open(uploaded_file).resize((32, 32))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Make predictions
        predictions = model.predict(image_array)
        predicted_class = classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display the result
        st.success(f"Prediction: {predicted_class} ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = classes[np.argmax(predictions)]

    # Display prediction
    st.write(f"**Predicted Class:** {predicted_class}")
