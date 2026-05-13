import ee
import os
import requests
import numpy as np
from tqdm import tqdm

# 1. Initialize GEE
# REPLACE WITH YOUR PROJECT ID
PROJECT_ID = 'deforestationproject' 

try:
    ee.Initialize(project=PROJECT_ID)
except:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

# 2. Configuration
ROI = ee.Geometry.Rectangle([-62.0, -9.0, -61.0, -8.0]) # Rondônia, Brazil
START_DATE = '2021-01-01'
END_DATE = '2021-12-31'
OUTPUT_DIR = 'data/raw'
NUM_IMAGES = 400

# Ensure folders exist
os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)

def get_rgb_and_mask(point):
    # Fetch Sentinel-2 Collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate(START_DATE, END_DATE) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .first() # Get the best single image for this point

    if not s2: return None, None

    # Fetch Forest Loss Mask (Hansen Global Forest Change)
    gfc = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
    loss_mask = gfc.select('loss').clip(point)

    # Visualization Parameters (Server-Side Processing)
    visualization = {
        'min': 0,
        'max': 3000,
        'bands': ['B4', 'B3', 'B2'], # Red, Green, Blue
    }
    # Create the 8-bit RGB image
    rgb_image = s2.visualize(**visualization)

    return rgb_image, loss_mask

def download_image(url, filepath):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        return True
    return False

# 3. Main Loop (THE FIX IS HERE)
print(f"🌍 Generating {NUM_IMAGES} random locations...")

# Generate random points on server side
random_points = ee.FeatureCollection.randomPoints(ROI, NUM_IMAGES)

# Pull just the coordinates (List of [lon, lat]) - This is small and safe
point_coords = random_points.geometry().coordinates().getInfo()

print(f"⬇️  Starting Download...")

count = 0
# Iterate through the coordinates we just fetched
for i, coord in enumerate(tqdm(point_coords)):
    lon, lat = coord
    
    # Create a 2.5km buffer around the point
    point_geom = ee.Geometry.Point([lon, lat]).buffer(1280).bounds()

    try:
        rgb, mask = get_rgb_and_mask(point_geom)
        if not rgb: continue

        # URLs
        img_url = rgb.getThumbURL({
            'dimensions': [256, 256],
            'format': 'png'
        })
        mask_url = mask.getThumbURL({
            'dimensions': [256, 256],
            'format': 'png',
            'min': 0,
            'max': 1,
            'palette': ['black', 'white']
        })

        # Filenames
        img_name = f"data/raw/images/tile_{count:04d}.png"
        mask_name = f"data/raw/masks/tile_{count:04d}.png"

        # Download
        if download_image(img_url, img_name) and download_image(mask_url, mask_name):
            count += 1

    except Exception as e:
        print(f"Skipped {i}: {e}")
        continue

print("✅ Download Complete! You now have clean, standard PNG images.")