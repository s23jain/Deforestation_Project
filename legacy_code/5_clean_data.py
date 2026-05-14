import os
import glob
import cv2
import numpy as np
import shutil

# --- CONFIGURATION ---
IMG_DIR = os.path.join('data', 'raw', 'images')
MASK_DIR = os.path.join('data', 'raw', 'masks')
TRASH_DIR = os.path.join('data', 'trash')
# [CHANGE] New Threshold: 0.80 (80%)
# If image is >80% empty, we trash it.
# This keeps the "68% bad" images (which have data) but kills the "99% bad" ones.
THRESHOLD = 0.80 

def main():
    print("--- Phase 5: Data Cleaning (Forgiving Mode) ---")
    
    os.makedirs(TRASH_DIR, exist_ok=True)
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, '*.tif')))
    
    if not img_files:
        print("ERROR: No images found! Did you restore them from 'data/trash'?")
        return

    print(f"Inspecting {len(img_files)} images...")
    deleted_count = 0
    
    for img_path in img_files:
        filename = os.path.basename(img_path)
        mask_name = filename.replace('s2_', 'mask_')
        mask_path = os.path.join(MASK_DIR, mask_name)
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None: continue
            
        # 1. Count Invalid (Inf/NaN) + Pure Zero pixels
        # Mask for non-finite (inf/nan)
        bad_mask = ~np.isfinite(img)
        # Mask for pure zero
        zero_mask = (np.nan_to_num(img, nan=1.0) == 0)
        
        # Combine (logical OR)
        total_bad_pixels = np.sum(bad_mask | zero_mask)
        total_pixels = img.size
        
        bad_ratio = total_bad_pixels / total_pixels
        
        # --- DECISION ---
        if bad_ratio > THRESHOLD:
            print(f"REJECT: {filename} ({bad_ratio*100:.1f}% empty) -> TRASHED")
            
            # Move to Trash
            try:
                shutil.move(img_path, os.path.join(TRASH_DIR, filename))
                if os.path.exists(mask_path):
                    shutil.move(mask_path, os.path.join(TRASH_DIR, mask_name))
                deleted_count += 1
            except Exception as e:
                print(f"Error moving {filename}: {e}")
        else:
            # Optional: Print keep message for sanity check
            # print(f"KEEP:   {filename} ({bad_ratio*100:.1f}% empty)")
            pass

    print("-" * 30)
    print(f"Cleaning Complete.")
    print(f"Removed {deleted_count} useless images.")
    print(f"Retained {len(img_files) - deleted_count} valid images.")

if __name__ == "__main__":
    main()