import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

# Load the model (assuming the model is in the .pkl format)
def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_from_json(model_data["architecture"])
    model.set_weights(model_data["weights"])
    return model

# Load the blood cell classification model
model = load_model("blood_cells_classification.pkl")

# Streamlit UI
st.set_page_config(page_title="Blood Cell Classification", page_icon="🩸", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f0f0f0;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stSelectbox {color: black;}
    .stImage {border-radius: 10px;}
    .stProgress > div > div {background-color: #4CAF50;}
    </style>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/White_Blood_Cells_35x_%28cropped%29.jpg/640px-White_Blood_Cells_35x_%28cropped%29.jpg", width=200)
st.sidebar.title("🩺 Blood Cell Classifier")
st.sidebar.write("This tool uses **Deep Learning** to classify blood cell types based on an image of the cell.")

# Main Section
st.title("🩸 Blood Cell Classification")
st.write("Upload an image of a blood cell to classify its type using a deep learning model.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224)).convert('RGB')  # Resize and convert to RGB
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Prediction progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class
    confidence = np.max(prediction, axis=1)[0]  # Get the confidence level

    # Display results
    classes = ['Eosinophil', 'Basophil', 'Neutrophil', 'Lymphocyte', 'Monocyte']  # Example classes
    predicted_label = classes[predicted_class]  # Convert the predicted class index to label
    st.success(f"✅ **Predicted Class: {predicted_label}**")
    st.write(f"🧪 **Confidence Score:** `{confidence:.4f}`")

    # Extra message based on confidence
    if confidence > 0.8:
        st.info(f"✅ The prediction is quite confident. The cell is classified as **{predicted_label}**.")
    else:
        st.warning("⚠️ The model is less confident in this prediction. Please consult a specialist for confirmation.")

