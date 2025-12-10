import cv2
import numpy as np
import glob

mask_path = glob.glob("data/raw/masks/*.tif")[0] # Get first mask
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

print(f"Max value in mask: {np.max(mask)}")
print(f"Number of forest pixels: {np.sum(mask == 1)}")