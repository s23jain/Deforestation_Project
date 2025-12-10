import ee
import geemap
import os

# --- 1. CONFIGURATION ---
MY_PROJECT = 'deforestationproject' 

OUTPUT_DIR = os.path.join('data', 'raw')
IMG_DIR = os.path.join(OUTPUT_DIR, 'images')
MASK_DIR = os.path.join(OUTPUT_DIR, 'masks')

START_DATE = '2023-01-01'
END_DATE = '2023-12-31'

# [NEW] Quality Threshold
# We will only download images that are at least 90% full of valid data.
# This prevents the "Black Bar" issue.
VALID_DATA_THRESHOLD = 0.90 

def main():
    # --- 2. INITIALIZATION ---
    print(f"Connecting to Earth Engine Project: {MY_PROJECT}...")
    try:
        ee.Initialize(project=MY_PROJECT)
    except Exception as e:
        print("Authentication required...")
        ee.Authenticate()
        ee.Initialize(project=MY_PROJECT)
    
    print("Google Earth Engine initialized successfully!")

    # --- 3. DEFINE ROI & GRID ---
    # We slightly shift the ROI to a denser part of the Amazon to ensure better data coverage
    ROI = ee.Geometry.Rectangle([-62.0, -9.0, -61.0, -8.0]) 

    # We use a 20x20 grid (400 tiles)
    rows, cols = 20, 20 
    print(f"Generating {rows}x{cols} tile grid...")
    grid = geemap.fishnet(ROI, rows=rows, cols=cols)
    tiles = grid.getInfo()['features'] 
    total_tiles = len(tiles)
    print(f"Total grid slots: {total_tiles}")

    # --- 4. PREPARE DIRECTORIES ---
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    # --- 5. PREPARE LAYERS ---
    print("Preparing layers...")
    
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)

    # We use the median composite which helps remove clouds, 
    # but edges might still be empty if the satellite didn't visit enough.
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(START_DATE, END_DATE) \
        .filterBounds(ROI) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
        .map(mask_s2_clouds) \
        .median() \
        .select(['B4', 'B3', 'B2']) \
        .clip(ROI)

    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(ROI)
    forest_mask = worldcover.eq(10) # 10 = Trees

    # --- 6. SMART DOWNLOAD LOOP ---
    print(f"Starting Smart Download (Filtering >{VALID_DATA_THRESHOLD*100}% valid data)...")
    
    downloaded_count = 0

    for i, tile in enumerate(tiles):
        roi_geometry = ee.Geometry(tile['geometry'])
        
        # --- [NEW] SERVER-SIDE QUALITY CHECK ---
        # 1. We look at the 'B4' (Red) band.
        # 2. We count how many pixels are NOT masked (valid).
        # 3. We use a coarse scale (100m) for this check to make it super fast.
        
        # Create a binary image: 1 if valid data, 0 if masked/empty
        valid_mask = s2.select('B4').mask()
        
        # Calculate the mean of this mask (0.0 to 1.0)
        # 1.0 means 100% valid data, 0.5 means 50% missing (black bars)
        stats = valid_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi_geometry,
            scale=100, # Fast check resolution
            maxPixels=1e9
        )
        
        # Get the result from the server
        valid_fraction = stats.get('B4').getInfo()

        # SKIP if the image is too empty
        if valid_fraction is None or valid_fraction < VALID_DATA_THRESHOLD:
            print(f"[{i+1}/{total_tiles}] Skipped: Low Quality ({valid_fraction:.2%} valid)")
            continue

        # --- DOWNLOAD IF GOOD ---
        img_name = f"s2_{i:04d}.tif"
        mask_name = f"mask_{i:04d}.tif"
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, mask_name)

        if os.path.exists(img_path) and os.path.exists(mask_path):
            print(f"[{i+1}/{total_tiles}] Skipping (Already exists)")
            downloaded_count += 1
            continue

        print(f"[{i+1}/{total_tiles}] Downloading... (Quality: {valid_fraction:.2%})")

        try:
            # Download Image
            geemap.download_ee_image(
                image=s2,
                filename=img_path,
                region=roi_geometry,
                crs='EPSG:4326',
                scale=10, # High res for actual download
                overwrite=True
            )

            # Download Mask
            geemap.download_ee_image(
                image=forest_mask,
                filename=mask_path,
                region=roi_geometry,
                crs='EPSG:4326',
                scale=10,
                overwrite=True
            )
            downloaded_count += 1
        except Exception as e:
            print(f"Error downloading tile {i}: {e}")

    print("-" * 30)
    print(f"Smart Download Complete.")
    print(f"Downloaded {downloaded_count} high-quality images out of {total_tiles} grid slots.")

if __name__ == "__main__":
    main()