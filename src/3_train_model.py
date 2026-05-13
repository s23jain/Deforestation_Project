import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import glob

# CONFIG
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 25  # Increased slightly

# --- CUSTOM LOSS FUNCTION (The Fix) ---
def weighted_binary_crossentropy(y_true, y_pred):
    """
    Forces the model to pay 10x more attention to white pixels (Deforestation).
    """
    # Calculate standard loss
    bce = K.binary_crossentropy(y_true, y_pred)
    
    # Create a weight map: 
    # If pixel is Deforestation (1), weight = 10. 
    # If pixel is Background (0), weight = 1.
    weight_vector = y_true * 9.0 + 1.0
    
    # Apply weights
    weighted_bce = weight_vector * bce
    
    return K.mean(weighted_bce)

def simple_unet_model(n_classes=1, img_height=256, img_width=256, img_channels=3):
    inputs = Input((img_height, img_width, img_channels))
    
    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    # COMPILE WITH CUSTOM LOSS
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss=weighted_binary_crossentropy, 
                  metrics=['accuracy'])
    return model

def load_data(path):
    images = sorted(glob.glob(os.path.join(path, "images/*.png")))
    masks = sorted(glob.glob(os.path.join(path, "masks/*.png")))
    
    X = []
    Y = []
    
    for img_path, mask_path in zip(images, masks):
        # Read & Normalize Image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = img / 255.0
        
        # Read & Normalize Mask (Binarize strict)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
        # Ensure mask is strictly 0 or 1
        mask = (mask > 128).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        X.append(img)
        Y.append(mask)
        
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    print("⏳ Loading Data...")
    X_train, Y_train = load_data('data/processed/train')
    X_val, Y_val = load_data('data/processed/val')
    print(f"✅ Loaded: {len(X_train)} training images")

    print("🏗️ Building U-Net with Weighted Loss...")
    model = simple_unet_model()
    
    print("🚀 Starting Aggressive Training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1
    )
    
    # Save (Make sure to save as .keras)
    model.save('models/unet_model.keras')
    print("💾 Aggressive Model saved!")