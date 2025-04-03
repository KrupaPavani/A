import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json

# Load the model
def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_from_json(model_data["architecture"])
    model.set_weights(model_data["weights"])
    return model

# Load blood cell classification model
model = load_model("blood_cells_classification.pkl")

# Function to detect purple-stained regions
def detect_purple_stain(image):
    image_np = np.array(image)
    
    # Extract RGB channels
    red, green, blue = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]
    
    # Condition: High Red & Blue, Low Green → Purple Stain
    purple_mask = (red > 100) & (blue > 100) & (green < 80)
    
    # If >10% of the image contains purple pixels, classify it as a blood cell image
    purple_ratio = np.sum(purple_mask) / purple_mask.size
    return purple_ratio > 0.1  # Adjust threshold as needed

# Streamlit UI
st.set_page_config(page_title="Blood Cell Classification", page_icon="🩸", layout="wide")
st.sidebar.title("🔬 Automated Blood Cell Classification")
st.sidebar.markdown("---")
st.sidebar.subheader("📌 About")
st.sidebar.info("This app classifies blood cells using deep learning. Upload a blood cell image, and the model will predict its type.")

# Main Section
st.title("🩸 Blood Cell Classification")
st.write("Upload an image of a blood cell to classify its type using a deep learning model.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Check if image contains purple stain
    if not detect_purple_stain(image):
        st.error("❌ Not a blood cell image. Please upload a valid blood cell image.")
    else:
        # Preprocess the image
        image = image.resize((224, 224)).convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        # Define blood cell classes
        blood_cell_classes = ['monocyte', 'platelet', 'lymphocyte', 'basophil', 'eosinophil', 'ig', 'neutrophil', 'erythroblast']

        # **Proceed only if the model is confident**
        if confidence > 0.9:
            predicted_label = blood_cell_classes[predicted_class]
            st.success(f"✅ **Predicted Class: {predicted_label}**")
            st.write(f"🧪 **Confidence Score:** `{confidence:.4f}`")
        else:
            st.warning("⚠️ Please provide a clearer image to identify the blood cell type.")
