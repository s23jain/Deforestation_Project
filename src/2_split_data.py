import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DIR = os.path.join('data', 'processed')
SPLIT_RATIO = 0.8  # 80% Training, 20% Validation

def main():
    print("--- Phase 2: Data Splitting & Visualization ---")
    
    # 1. Get Lists of Files
    # We sort them to ensure s2_001 always matches mask_001
    img_paths = sorted(glob.glob(os.path.join(DATA_DIR, 'images', '*.tif')))
    mask_paths = sorted(glob.glob(os.path.join(DATA_DIR, 'masks', '*.tif')))

    # Safety Check
    if len(img_paths) == 0:
        print("ERROR: No images found! Did you run step 1?")
        return
    if len(img_paths) != len(mask_paths):
        print(f"ERROR: Mismatch! Found {len(img_paths)} images and {len(mask_paths)} masks.")
        return

    print(f"Found {len(img_paths)} image pairs.")

    # 2. Shuffle and Split
    # Zip them together so the image always stays with its mask
    combined = list(zip(img_paths, mask_paths))
    random.seed(42) # Fixed seed for reproducibility (DVC loves this)
    random.shuffle(combined)

    split_idx = int(len(combined) * SPLIT_RATIO)
    train_pairs = combined[:split_idx]
    val_pairs = combined[split_idx:]

    print(f"Training Samples:   {len(train_pairs)}")
    print(f"Validation Samples: {len(val_pairs)}")

    # 3. Save Lists to Text Files
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    def save_list(pairs, filename):
        with open(os.path.join(PROCESSED_DIR, filename), 'w') as f:
            for img, mask in pairs:
                # We save relative paths to keep things clean
                f.write(f"{img},{mask}\n")
    
    save_list(train_pairs, 'train_list.txt')
    save_list(val_pairs, 'val_list.txt')
    print(f"Saved lists to {PROCESSED_DIR}")

    # 4. Generate a Preview (Sanity Check)
    print("Generating preview...")
    preview_idx = 0
    sample_img_path, sample_mask_path = train_pairs[preview_idx]

    # Read Image (Sentinel-2 is often 16-bit or Float, we convert to visible)
    img = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(sample_mask_path, cv2.IMREAD_UNCHANGED)

    # Normalize for display (Sentinel values are often 0-10000 or 0-1)
    # If max value > 1, assume it needs scaling
    if img.max() > 1.0:
        img = img / 255.0 # Simple scaling for preview
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Sentinel-2 Satellite Image")
    # Swap BGR (OpenCV standard) to RGB (Matplotlib standard)
    plt.imshow(img[:, :, ::-1]) 
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth (Forest Mask)")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'preview.png'))
    print(f"Preview saved to {os.path.join(PROCESSED_DIR, 'preview.png')}")
    print("Phase 2 Complete!")

if __name__ == "__main__":
    main()