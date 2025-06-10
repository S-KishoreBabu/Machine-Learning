import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model("../model/tomato_disease_model.h5")

# Class names (order matters!)
class_names = os.listdir("../dataset/")  # Make sure this matches your folder order
class_names.sort()

st.title("🍅 Tomato Disease Prediction")

uploaded_file = st.file_uploader("Upload a Tomato Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(128, 128))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🧠 Predicted Disease: **{predicted_class.replace('Tomato___', '').replace('_', ' ')}**")
