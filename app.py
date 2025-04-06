import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
import colorsys
import time

# Load the model
def load_model(pkl_path):
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_from_json(model_data["architecture"])
    model.set_weights(model_data["weights"])
    return model

# Load blood cell classification model
try:
    model = load_model("blood_cells_classification.pkl")
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# Detect purple stain in image
def detect_purple_stain(image):
    image_np = np.array(image)
    hsv_pixels = np.array([colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
                          for r, g, b in image_np.reshape(-1, 3)])
    hue, saturation, value = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    purple_mask = ((hue >= 0.6) & (hue <= 0.75)) & (saturation > 0.2) & (value > 0.2)
    purple_ratio = np.sum(purple_mask) / len(hue)
    return purple_ratio > 0.05  # At least 5% purple

# Streamlit UI
st.set_page_config(page_title="Blood Cell Classification", page_icon="🩸", layout="wide")
st.sidebar.title("🔬 Automated Blood Cell Classification")
st.sidebar.markdown("---")
st.sidebar.subheader("📌 About")
st.sidebar.info("This app classifies blood cells using deep learning. Upload a blood cell image, and the model will predict its type.")

st.title("🩸 Blood Cell Classification")
st.write("Upload an image of a blood cell to classify its type using a deep learning model.")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Check for purple stain
    if not detect_purple_stain(image):
        st.error("❌ Not a blood cell image. Please upload a valid blood cell image.")
    else:
        # Dynamically get input size from model and resize
        _, height, width, channels = model.input_shape
        image = image.resize((width, height)).convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Simulate progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.005)
            progress_bar.progress(percent_complete + 1)

        # Predict
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        # Class labels
        blood_cell_classes = ['monocyte', 'platelet', 'lymphocyte', 'basophil',
                              'eosinophil', 'ig', 'neutrophil', 'erythroblast']

        # Show result
        if confidence > 0.9:
            predicted_label = blood_cell_classes[predicted_class]
            st.success(f"✅ **Predicted Class: {predicted_label}**")
            st.write(f"🧪 **Confidence Score:** `{confidence:.4f}`")
        else:
            st.warning("⚠️ Please provide a clearer image to identify the blood cell type.")
