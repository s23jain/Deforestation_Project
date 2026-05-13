import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# CONFIG
RAW_IMG_DIR = 'data/raw/images'
RAW_MASK_DIR = 'data/raw/masks'
PROCESSED_DIR = 'data/processed'

def split_data():
    # 1. Gather all files (PNGs now!)
    files = [f for f in os.listdir(RAW_IMG_DIR) if f.endswith('.png')]
    
    if len(files) == 0:
        print("❌ Error: No PNG images found! Did the download work?")
        return

    print(f"📦 Found {len(files)} total images. Splitting...")

    # 2. Split (80% Train, 10% Val, 10% Test)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    # 3. Helper to move files
    def move_files(file_list, split_name):
        dest_img = os.path.join(PROCESSED_DIR, split_name, 'images')
        dest_mask = os.path.join(PROCESSED_DIR, split_name, 'masks')
        
        # Create folders if not exist
        os.makedirs(dest_img, exist_ok=True)
        os.makedirs(dest_mask, exist_ok=True)
        
        for f in file_list:
            # Copy Image
            shutil.copy(os.path.join(RAW_IMG_DIR, f), os.path.join(dest_img, f))
            # Copy Mask (Exact same filename)
            shutil.copy(os.path.join(RAW_MASK_DIR, f), os.path.join(dest_mask, f))

    # 4. Execute Move
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')
    
    print(f"✅ Data Split Complete!")
    print(f"   Train: {len(train_files)}")
    print(f"   Val:   {len(val_files)}")
    print(f"   Test:  {len(test_files)}")

if __name__ == "__main__":
    split_data()