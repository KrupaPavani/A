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
st.sidebar.title(" An Automated Blood Cell Classification")
st.sidebar.markdown("---")
st.sidebar.subheader("📌 About")
st.sidebar.info(
    "This app classifies blood cells using a deep learning model. "
    "Upload a blood cell image, and the model will predict its type.")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f0f0f0;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stSelectbox {color: black;}
    .stImage {border-radius: 10px; max-width: 300px;}
    .stProgress > div > div {background-color: #4CAF50;}
    </style>
    """, unsafe_allow_html=True)

# Main Section
st.title("🩸 Blood Cell Classification")
st.write("Upload an image of a blood cell to classify its type using a deep learning model.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=False, width=300)

    # Preprocess the image
    image = image.resize((224, 224)).convert('RGB')  # Resize to model's expected input size
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Debugging: Display image shape
    st.write(f"🔍 Image array shape: {image_array.shape}")

    # Prediction progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)

    # Make prediction
    try:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class
        confidence = np.max(prediction, axis=1)[0]  # Get the confidence level

        # Define blood cell classes
        blood_cell_classes = ['monocyte', 'platelet', 'lymphocyte', 'basophil', 'eosinophil', 'ig', 'neutrophil', 'erythroblast']
        
        # Check if the predicted class is a blood cell
        if predicted_class < len(blood_cell_classes):
            predicted_label = blood_cell_classes[predicted_class]  # Convert the predicted class index to label
            if confidence > 0.9:
                st.success(f"✅ **Predicted Class: {predicted_label}**")
                st.write(f"🧪 **Confidence Score:** `{confidence:.4f}`")
            else:
                st.warning("⚠️ Please provide a clearer image to identify the blood cell type.")
        else:
            st.error("❌ Not a blood cell image. Please provide a valid blood cell image.")
    except ValueError as e:
        st.error(f"🚨 Model input error: {str(e)}. Please ensure the uploaded image is valid.")
