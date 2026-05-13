import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import cv2
import glob

# CONFIG
IMG_HEIGHT = 256
IMG_WIDTH  = 256
BATCH_SIZE = 8
EPOCHS     = 50

# --- DICE LOSS ---
# Dice Loss directly optimizes mask overlap. It's immune to the all-black /
# all-white collapse that weighted BCE suffers from, because it compares the
# predicted mask shape against the true mask shape regardless of class ratio.
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# --- BCE + DICE COMBINED LOSS ---
# BCE keeps training stable in early epochs (gives gradient everywhere).
# Dice drives the model to actually match mask shapes.
# Together they avoid both the "all-black" and "all-white" collapse.
def bce_dice_loss(y_true, y_pred):
    bce  = K.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return K.mean(bce) + dice

# --- IoU METRIC ---
# The honest measure of segmentation quality.
# 0.0 = predicting all one class. >0.2 = genuinely learning.
def iou_metric(y_true, y_pred):
    y_pred        = K.cast(y_pred > 0.5, dtype='float32')
    intersection  = K.sum(y_true * y_pred)
    union         = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def simple_unet_model(n_classes=1, img_height=256, img_width=256, img_channels=3):
    inputs = Input((img_height, img_width, img_channels))

    # Encoder — added BatchNorm after each block for training stability
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck + Dropout to prevent overfitting on small dataset
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)

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
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=bce_dice_loss,
        metrics=['accuracy', iou_metric]
    )
    return model

def load_data(path):
    images = sorted(glob.glob(os.path.join(path, "images/*.png")))
    masks  = sorted(glob.glob(os.path.join(path, "masks/*.png")))

    if len(images) == 0:
        raise FileNotFoundError(f"No images found in {path}/images/")
    if len(images) != len(masks):
        raise ValueError(f"Mismatch: {len(images)} images vs {len(masks)} masks in {path}")

    X, Y = [], []

    for img_path, mask_path in zip(images, masks):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read: {img_path}, skipping.")
            continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Could not read: {mask_path}, skipping.")
            continue
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        mask = (mask > 128).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        Y.append(mask)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

if __name__ == "__main__":
    print("Loading Data...")
    X_train, Y_train = load_data('data/processed/train')
    X_val,   Y_val   = load_data('data/processed/val')
    print(f"Train: {len(X_train)} images  |  Val: {len(X_val)} images")

    deforested_pct = Y_train.mean() * 100
    print(f"Class balance: {deforested_pct:.2f}% deforested pixels in training masks")

    print("Building U-Net...")
    model = simple_unet_model()
    model.summary()

    # --- THE FIX: Callbacks now monitor 'val_iou_metric' and look for the 'max' value ---
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_iou_metric',
            mode='max', # Higher IoU is better
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_iou_metric',
            mode='max', # Higher IoU is better
            patience=12, # Giving it a bit more time to learn
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("Starting Training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/unet_model.keras')
    print("Model saved to models/unet_model.keras")

    final_iou  = history.history.get('val_iou_metric', [0])[-1]
    final_loss = history.history.get('val_loss', [0])[-1]
    print(f"\nFinal Results:")
    print(f"   Val Loss : {final_loss:.4f}")
    print(f"   Val IoU  : {final_iou:.4f}")
    if final_iou < 0.05:
        print("   WARNING: IoU near 0 — model predicting one class only.")
    elif final_iou < 0.15:
        print("   Model is learning but weakly. Try running app.py with sensitivity at 0.2.")
    else:
        print("   IoU looks healthy — model is detecting deforestation.")