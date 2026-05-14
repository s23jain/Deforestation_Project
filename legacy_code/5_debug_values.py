import os
import cv2
import numpy as np
import glob

# --- CONFIG ---
IMG_DIR = os.path.join('data', 'raw', 'images')

def main():
    print("--- Data Inspector ---")
    # Get a few random images
    files = glob.glob(os.path.join(IMG_DIR, '*.tif'))[:5]
    
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None: continue
        
        # Calculate stats
        min_val = np.nanmin(img)
        max_val = np.nanmax(img)
        mean_val = np.nanmean(img)
        
        print(f"File: {os.path.basename(f)}")
        print(f"  Min: {min_val:.5f} | Max: {max_val:.5f} | Mean: {mean_val:.5f}")
        
        # Check for NaN (Corruption)
        if np.isnan(img).any():
            print("  [WARNING] Contains NaN values!")
            
    print("\nIf Mean is < 0.1, the image is too dark for the AI.")

if __name__ == "__main__":
    main()