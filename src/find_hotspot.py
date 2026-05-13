import os
import cv2
import numpy as np
import glob

# CONFIG
MASK_DIR = 'data/raw/masks'

def find_deforested_image():
    print("🔍 Scanning dataset for heavy deforestation...")
    
    # Get all mask files
    mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))
    
    if not mask_files:
        print("❌ No masks found! Check your data folder.")
        return

    # Check each file
    for mask_path in mask_files:
        # Read as Grayscale (0=Black, 255=White)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate percentage of white pixels
        white_pixels = np.sum(mask > 128)
        total_pixels = 256 * 256
        ratio = white_pixels / total_pixels
        
        # If more than 10% is deforested, stop and show us
        if ratio > 0.10: 
            filename = os.path.basename(mask_path)
            print(f"\n🔥 FOUND ONE!")
            print(f"📂 File: {filename}")
            print(f"🌲 Deforestation Level: {ratio:.1%}")
            print(f"👉 Use this file to test your model!")
            return

    print("⚠️ No heavily deforested images found in this batch.")

if __name__ == "__main__":
    find_deforested_image()