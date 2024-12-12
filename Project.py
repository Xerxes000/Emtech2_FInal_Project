import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("mnist_model.h5")

# Title of the app
st.title("MNIST Digit Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of a digit (28x28 pixels)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Display the results
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
