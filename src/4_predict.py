import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU mode

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random

# CONFIG
MODEL_PATH = 'models/unet_model.keras'
IMAGE_DIR = 'data/raw/images'  # Where your new PNGs are
OUTPUT_DIR = 'results'

def predict_and_viz():
    # 1. Load Model
    print(f"⏳ Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Get a random PNG image
    # We look for .png now, not .tif
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not image_paths:
        print(f"❌ No PNG images found in {IMAGE_DIR}")
        return
        
    # Pick a random file to test
    img_path = random.choice(image_paths)
    print(f"🔍 Analyzing: {os.path.basename(img_path)}")

    # 3. Preprocess (The "Visual" Way)
    # Load as standard RGB image
    original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize to 256x256 (Model input size)
    input_img = cv2.resize(original_img, (256, 256))
    
    # Normalize (Simple division by 255 because it's a PNG)
    input_tensor = input_img.astype(np.float32) / 255.0
    
    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 4. Predict
    print("🚀 Running prediction...")
    prediction = model.predict(input_tensor)[0] # Shape (256, 256, 1)
    
    # Threshold (Confidence > 0.5 = Deforestation)
    mask = (prediction > 0.5).astype(np.uint8)

    # 5. Visualize
    plt.figure(figsize=(12, 4))

    # Plot 1: Input Image
    plt.subplot(1, 3, 1)
    plt.title("Satellite Input")
    plt.imshow(input_img)
    plt.axis('off')

    # Plot 2: Model Confidence (Heatmap)
    plt.subplot(1, 3, 2)
    plt.title("AI Confidence Map")
    plt.imshow(prediction, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    # Plot 3: Final Mask (Black/White)
    plt.subplot(1, 3, 3)
    plt.title("Deforestation Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    # Save result
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "latest_prediction.png")
    plt.savefig(save_path)
    print(f"💾 Result saved to: {save_path}")
    
    # Show it (if you are on a local machine)
    # plt.show()

if __name__ == "__main__":
    predict_and_viz()