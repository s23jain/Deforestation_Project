import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# --- CONFIGURATION ---
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4

TRAIN_LIST = os.path.join('data', 'processed', 'train_list.txt')
VAL_LIST = os.path.join('data', 'processed', 'val_list.txt')
MODEL_SAVE_PATH = os.path.join('models', 'unet_model.keras')

# --- 1. DATA GENERATOR (CLIPPED VERSION) ---
class DeforestationDataGen(tf.keras.utils.Sequence):
    def __init__(self, list_path, batch_size=8, img_size=256):
        with open(list_path, 'r') as f:
            self.pairs = [line.strip().split(',') for line in f.readlines()]
        self.batch_size = batch_size
        self.img_size = img_size
        
    def __len__(self):
        return len(self.pairs) // self.batch_size

    def __getitem__(self, index):
        batch_pairs = self.pairs[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        masks = []
        
        for img_path, mask_path in batch_pairs:
            if not os.path.exists(img_path):
                img_path = os.path.join(os.getcwd(), img_path)
                mask_path = os.path.join(os.getcwd(), mask_path)

            # --- PROCESS IMAGE ---
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None: continue
            
            # 1. Force Resize
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.astype(np.float32)

            # 2. [CRITICAL FIX] Clip Values
            # This forces the -7e37 values to become exactly 0.0
            # And assumes max value should be roughly 255.0 or 1.0
            if img.max() > 1.0:
                img = img / 255.0
            
            # Final Safety Clip: Ensure strictly 0.0 to 1.0
            img = np.clip(img, 0.0, 1.0)
            
            # --- PROCESS MASK ---
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)
            
        return np.array(images), np.array(masks)

# --- 2. U-NET MODEL ---
def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    return models.Model(inputs=[inputs], outputs=[outputs])

# --- 3. TRAINING LOOP ---
def main():
    print("--- Phase 3: Model Training ---")
    
    train_gen = DeforestationDataGen(TRAIN_LIST, batch_size=BATCH_SIZE)
    val_gen = DeforestationDataGen(VAL_LIST, batch_size=BATCH_SIZE)
    
    # Sanity Check
    X_sample, y_sample = train_gen.__getitem__(0)
    print(f"Sanity Check - Input Range: {X_sample.min()} to {X_sample.max()}")
    
    # Build
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, 
                                           monitor='val_loss', 
                                           save_best_only=True,
                                           mode='min',
                                           verbose=1)
    
    print("Starting Training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )
    
    print("Training Complete!")

if __name__ == "__main__":
    main()