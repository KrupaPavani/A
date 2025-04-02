import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# Streamlit UI
st.title("Blood Cell Classification")
st.write("Upload an image to classify the type of blood cell.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Send image to backend API (Colab)
    backend_url = "https://colab.research.google.com/drive/1jqjmvg-PdVbzVx52j0OBO02G6bsEmrqB"  # Colab backend URL
    files = {'file': ('image.png', img_bytes, 'image/png')}
    response = requests.post(backend_url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        predicted_class = result.get("class", "Unknown")
        confidence = result.get("confidence", 0) * 100
        
        # Display result
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
    else:
        st.error("Failed to get a response from the backend. Please try again.")
