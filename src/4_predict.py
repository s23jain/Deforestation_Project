import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import glob
import random

# CONFIG
MODEL_PATH = 'models/unet_model.keras'
IMAGE_DIR  = 'data/raw/images'
OUTPUT_DIR = 'results'


# Must match the function used at training time exactly
def weighted_binary_crossentropy(y_true, y_pred):
    bce = K.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * 9.0 + 1.0
    return K.mean(weight_vector * bce)


def predict_and_viz():
    print(f"⏳ Loading model from {MODEL_PATH}...")
    try:
        # FIX: plain load_model() crashes here because the model was saved with a
        # custom loss. Must pass custom_objects so Keras can deserialize it.
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy}
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

    # Load as BGR, convert to RGB (consistent with training)
    original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    input_img = cv2.resize(original_img, (256, 256))
    input_tensor = input_img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, 256, 256, 3)

    print("🚀 Running prediction...")
    prediction = model.predict(input_tensor)[0]  # (256, 256, 1)
    mask = (prediction > 0.5).astype(np.uint8)

    coverage = mask.mean() * 100
    print(f"🌲 Deforestation coverage: {coverage:.2f}%")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Satellite Input")
    plt.imshow(input_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("AI Confidence Map")
    plt.imshow(prediction[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Deforestation Mask")
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.axis('off')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "latest_prediction.png")
    plt.savefig(save_path)
    print(f"💾 Result saved to: {save_path}")

if __name__ == "__main__":
    predict_and_viz()