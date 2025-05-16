import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('../model/tomato_disease_model.h5')

# Class labels (edit based on your dataset folders)
class_names = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold']  # Add all used classes

st.title("🍅 Tomato Leaf Disease Prediction")

uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {np.max(prediction)*100:.2f}%")
