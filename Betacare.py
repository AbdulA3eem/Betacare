#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your pre-trained model
model = tf.keras.models.load_model('modelb.h5')  # Update this with your model's path

# Function to preprocess the image and make a prediction
def preprocess_and_predict(image, model):
    # Resize the image to the required input size for your model (256x256 in your case)
    image = image.resize((256, 256))
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Normalize the image (as you did during training)
    image_array = image_array / 255.0

    # Expand dimensions to match the input format of the model
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Get predictions from the model
    predictions = model.predict(image_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return predicted_class

# Streamlit app layout
st.title("Diabetic Retinopathy Detection")
st.write("Upload a retina scan image, and the model will predict the severity of diabetic retinopathy.")

# File uploader for retina scan image
uploaded_file = st.file_uploader("Choose a retina scan image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Retina Scan', use_column_width=True)
    st.write("Classifying...")

    # Predict the diagnosis using the model
    prediction = preprocess_and_predict(image, model)

    # Map the prediction to the corresponding label
    labels = {
        0: "No Diabetic Retinopathy (No DR)",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative Diabetic Retinopathy"
    }
    
    # Display the prediction
    st.write(f"Prediction: {labels[prediction]}")

