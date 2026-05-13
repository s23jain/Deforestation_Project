import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU mode

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Amazon Deforestation Detector", page_icon="🌲", layout="wide")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Define all custom functions used during training so Keras can load them
    def dice_loss(y_true, y_pred):
        import tensorflow.keras.backend as K
        smooth = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1.0 - (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def bce_dice_loss(y_true, y_pred):
        import tensorflow.keras.backend as K
        bce = K.binary_crossentropy(y_true, y_pred)
        return K.mean(bce) + dice_loss(y_true, y_pred)

    def iou_metric(y_true, y_pred):
        import tensorflow.keras.backend as K
        y_pred = K.cast(y_pred > 0.5, dtype='float32')
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        return (intersection + 1e-7) / (union + 1e-7)

    try:
        return tf.keras.models.load_model(
            'models/unet_model.keras', 
            custom_objects={
                'bce_dice_loss': bce_dice_loss, 
                'dice_loss': dice_loss,
                'iou_metric': iou_metric
            }
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- HEADER ---
st.title("🛰️ Deforestation Detection from Space")
st.markdown("**Project Status:** 🟢 MVP Complete | Model includes X-Ray Confidence Heatmap")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Control Panel")
    st.info("Upload a .png file from 'data/raw/images'")
    # Fine-tuned slider to catch the exact threshold. Defaulting to 0.15 based on training logs!
    sensitivity = st.slider("Model Sensitivity (Threshold)", 0.0, 1.0, 0.15, 0.01)

# --- PROCESSING FUNCTION ---
def process_image(uploaded_file):
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
    original_img, input_img = process_image(uploaded_file)
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Scanning forest canopy..."):
            
            # Predict
            input_tensor = cv2.resize(input_img, (256, 256))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            pred = model.predict(input_tensor)[0]
            
            # Create Final Mask based on slider
            mask = (pred > sensitivity).astype(np.float32)
            mask_display = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
            coverage = (np.sum(mask) / (256*256)) * 100

            # --- 3-COLUMN DISPLAY ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("1. Satellite Input")
                st.image(original_img, use_container_width=True)

            with col2:
                st.subheader("2. AI Heatmap")
                # Generate a color-mapped visual of the AI's raw confidence
                fig, ax = plt.subplots(figsize=(5, 5))
                # Auto-scaling colorbar so you can see the shapes!
                im = ax.imshow(pred, cmap='jet')
                fig.colorbar(im, fraction=0.046, pad=0.04)
                ax.axis('off')
                st.pyplot(fig)

            with col3:
                st.subheader(f"3. Mask (Threshold: {sensitivity:.2f})")
                st.image(mask_display, use_container_width=True)
                
            # --- METRICS ---
            st.divider()
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Deforestation Coverage", f"{coverage:.2f}%")
            metric_col2.metric("AI Max Confidence", f"{pred.max():.3f}")
            metric_col3.metric("AI Min Confidence", f"{pred.min():.3f}")