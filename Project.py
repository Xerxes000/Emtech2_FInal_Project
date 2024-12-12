import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive File ID
MODEL_FILE_ID = "1bKAnp_btGozeaQqJbfHkE6-AePZOcYT7"
MODEL_PATH = "mobilenet_cifar10_model.h5"

# Download the model from Google Drive if not already downloaded
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... This may take a few minutes."):
        url = f"https://drive.google.com/file/d/1bKAnp_btGozeaQqJbfHkE6-AePZOcYT7/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# CIFAR-10 class names
classes = [
    "Airplane", "Automobile", "Bird", "Cat", 
    "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
]

# App title
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load and preprocess the image
    image = Image.open(uploaded_file).resize((32, 32))
    image_array = img_to_array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = classes[np.argmax(predictions)]

    # Display prediction
    st.write(f"**Predicted Class:** {predicted_class}")
