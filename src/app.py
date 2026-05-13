import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU mode

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Amazon Deforestation Detector", page_icon="🌲", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Define the custom loss so Keras recognizes it when loading
    def weighted_binary_crossentropy(y_true, y_pred):
        import tensorflow.keras.backend as K
        bce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * 9.0 + 1.0
        return K.mean(weight_vector * bce)

    try:
        return tf.keras.models.load_model(
            'models/unet_model.keras', 
            custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- HEADER ---
st.title("🛰️ Deforestation Detection from Space")
st.markdown("""
**Project Status:** 🟢 MVP Complete  
This tool uses a **U-Net Deep Learning Model** to analyze Sentinel-2 satellite imagery and detect deforestation patterns in the Amazon Rainforest.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    st.info("Upload a .png file from 'data/raw/images'")
    # You can drag this slider down to 0.2 or 0.1 if the model is still being shy!
    sensitivity = st.slider("Model Sensitivity", 0.0, 1.0, 0.5, 0.05)

# --- PROCESSING FUNCTION ---
# --- PROCESSING FUNCTION ---
def process_image(uploaded_file):
    """
    Standard Image Processing for RGB PNGs/JPGs
    """
    # 1. Read file as standard RGB image (For the UI Display)
    image = Image.open(uploaded_file).convert('RGB')
    image = np.array(image)
    
    # 2. THE FIX: Convert to BGR for the model input 
    # (Because the model was trained using OpenCV's BGR format)
    model_input_image = image[..., ::-1] 
    
    # 3. Normalize (0-255 -> 0.0-1.0)
    norm_image = model_input_image.astype(np.float32) / 255.0
    
    return image, norm_image

# --- MAIN APP ---
uploaded_file = st.file_uploader("Choose a satellite image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # 1. Process
    original_img, input_img = process_image(uploaded_file)
    
    # 2. Display Input
    with col1:
        st.subheader("1. Satellite Input")
        st.image(original_img, caption="Sentinel-2 Imagery", use_container_width=True)

    # 3. Predict
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Scanning forest canopy..."):
            # Resize to 256x256 (Model Requirement)
            input_tensor = cv2.resize(input_img, (256, 256))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # Predict
            pred = model.predict(input_tensor)[0]
            
            # Threshold
            mask = (pred > sensitivity).astype(np.float32)
            
            # Resize mask back to original size for display (optional, but looks nice)
            mask_display = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

            # Calculate Coverage
            coverage = (np.sum(mask) / (256*256)) * 100

        # 4. Display Output
        with col2:
            st.subheader("2. AI Analysis")
            st.image(mask_display, caption="Deforestation Mask (White = Detected)", use_container_width=True)
        
        # 5. Metrics
        st.divider()
        st.metric("Deforestation Coverage", f"{coverage:.2f}%")
        
        if coverage > 15:
            st.error("⚠️ Critical Warning: High level of deforestation detected.")
        elif coverage > 5:
            st.warning("⚠️ Warning: Moderate deforestation activity.")
        else:
            st.success("✅ Status: Healthy Forest (Low activity detected).")