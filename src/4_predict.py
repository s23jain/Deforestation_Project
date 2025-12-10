import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --- CONFIGURATION ---
IMG_SIZE = 256
VAL_LIST = os.path.join('data', 'processed', 'val_list.txt')
MODEL_PATH = os.path.join('models', 'unet_model.keras')
OUTPUT_IMG = os.path.join('data', 'processed', 'prediction_results.png')

def process_image(img_path):
    """Preprocess image EXACTLY like training"""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    
    # [CRITICAL] Apply the same fixes as training
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = img * 3.0  # Brightness Boost
    img = np.clip(img, 0.0, 1.0)
    
    return img

def process_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.float32)

def main():
    print("--- Phase 4: Prediction & Visualization ---")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found!")
        return
    
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(VAL_LIST, 'r') as f:
        lines = f.readlines()
    
    # Pick 5 random samples
    indices = np.random.choice(len(lines), 5, replace=False)
    samples = [lines[i].strip().split(',') for i in indices]

    plt.figure(figsize=(15, 15))
    
    for i, (img_rel_path, mask_rel_path) in enumerate(samples):
        # Fix paths
        img_path = os.path.join(os.getcwd(), img_rel_path)
        mask_path = os.path.join(os.getcwd(), mask_rel_path)

        original_img = process_image(img_path)
        true_mask = process_mask(mask_path)
        
        # Predict
        input_tensor = np.expand_dims(original_img, axis=0)
        pred_mask = model.predict(input_tensor)[0]
        
        # Threshold: Confidence > 50% = Forest
        pred_mask = (pred_mask > 0.5).astype(np.float32)

        # Plot
        # 1. Satellite
        plt.subplot(5, 3, i*3 + 1)
        if i == 0: plt.title("Satellite Input")
        plt.imshow(original_img[:, :, ::-1]) # BGR to RGB
        plt.axis('off')

        # 2. Ground Truth
        plt.subplot(5, 3, i*3 + 2)
        if i == 0: plt.title("Ground Truth")
        plt.imshow(true_mask, cmap='gray')
        plt.axis('off')

        # 3. AI Prediction
        plt.subplot(5, 3, i*3 + 3)
        if i == 0: plt.title("AI Prediction")
        plt.imshow(pred_mask, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Results saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()