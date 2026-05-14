import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import random

# CONFIG
MODEL_PATH = 'models/unet_model.keras'
IMAGE_DIR  = 'data/raw/images'
OUTPUT_DIR = 'results'

def predict_and_viz():
    print(f"⏳ Loading model from {MODEL_PATH}...")
    try:
        # FIX: Updated to match the new bce_dice_loss and iou_metric from training!
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

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'bce_dice_loss': bce_dice_loss, 
                'dice_loss': dice_loss,
                'iou_metric': iou_metric
            }
        )
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    if not image_paths:
        print(f"❌ No PNG images found in {IMAGE_DIR}")
        return

    img_path = random.choice(image_paths)
    print(f"🔍 Analyzing: {os.path.basename(img_path)}")

    # Load as RGB to match Streamlit App
    original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # BGR fix for model input (matching our app.py breakthrough)
    model_input = original_img[..., ::-1]
    
    input_img = cv2.resize(model_input, (256, 256))
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 256, 256, 3)

    print("🚀 Running prediction...")
    prediction = model.predict(input_tensor)[0]  # (256, 256, 1)
    
    # Using our calculated safe threshold from the dashboard!
    THRESHOLD = 0.15 
    mask = (prediction > THRESHOLD).astype(np.uint8)

    coverage = mask.mean() * 100
    print(f"🌲 Deforestation coverage: {coverage:.2f}%")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Satellite Input")
    plt.imshow(cv2.resize(original_img, (256, 256))) # Show RGB to user
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("AI Confidence Map")
    plt.imshow(prediction[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Deforestation Mask (T={THRESHOLD})")
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.axis('off')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "latest_prediction.png")
    plt.savefig(save_path)
    print(f"💾 Result saved to: {save_path}")

if __name__ == "__main__":
    predict_and_viz()