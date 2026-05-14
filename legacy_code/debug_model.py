import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU mode

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP
IMG_PATH = "data/raw/images/s2_0007.tif"  # Pick a file you know exists
MODEL_PATH = "models/unet_model.keras"

def main():
    print(f"🔍 Testing Model: {MODEL_PATH}")
    print(f"🖼️ Input Image: {IMG_PATH}")

    # 2. LOAD MODEL
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model Failed to Load: {e}")
        return

    # 3. LOAD IMAGE (The "Training" Way)
    # We use cv2.imread directly from disk, which we KNOW works.
    image = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print("❌ Error: Could not read image. Check the path.")
        return

    print(f"   Original Shape: {image.shape}")
    print(f"   Original Max Value: {image.max()}")

    # 4. PREPROCESS
    # Handle Channels (BGR -> RGB)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize (Crucial Step)
    if image.max() > 255:
        print("   Detected 16-bit image. Normalizing by / 65535.0")
        input_img = image.astype(np.float32) / 65535.0
    else:
        print("   Detected 8-bit image. Normalizing by / 255.0")
        input_img = image.astype(np.float32) / 255.0

    # Resize
    input_img = cv2.resize(input_img, (256, 256))
    input_tensor = np.expand_dims(input_img, axis=0) # Shape: (1, 256, 256, 3)

    # 5. PREDICT
    print("🚀 Running Prediction...")
    pred = model.predict(input_tensor)[0]
    
    # Threshold
    mask = (pred > 0.5).astype(np.float32)
    
    # 6. VISUALIZE
    print("💾 Saving debug result to 'debug_output.png'...")
    
    plt.figure(figsize=(12, 4))
    
    # Input (Brightened for display)
    plt.subplot(1, 3, 1)
    plt.title(f"Input (Max: {input_img.max():.4f})")
    plt.imshow(np.clip(input_img * 3.5, 0, 1)) # Brighten x3.5
    
    # Raw Prediction (Confidence)
    plt.subplot(1, 3, 2)
    plt.title("Raw Model Confidence")
    plt.imshow(pred, cmap='jet')
    plt.colorbar()
    
    # Final Mask
    plt.subplot(1, 3, 3)
    plt.title("Final Binary Mask")
    plt.imshow(mask, cmap='gray')
    
    plt.savefig("debug_output.png")
    print("✅ Done! Check 'debug_output.png'.")

if __name__ == "__main__":
    main()