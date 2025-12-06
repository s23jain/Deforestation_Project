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
    ROI = ee.Geometry.Rectangle([-62.0, -9.0, -61.0, -8.0]) 

    # Create the grid
    # We use a smaller grid (20x20) for testing speed, giving ~400 images.
    # You can change this back to 40x40 later if you want more data.
    rows, cols = 20, 20 
    print(f"Generating {rows}x{cols} tile grid...")
    grid = geemap.fishnet(ROI, rows=rows, cols=cols)
    
    # Convert the grid to a client-side list so we can loop in Python
    print("Fetching grid metadata...")
    tiles = grid.getInfo()['features'] 
    total_tiles = len(tiles)
    print(f"Total tiles to process: {total_tiles}")

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

    # --- 6. MANUAL DOWNLOAD LOOP ---
    print("Starting manual download loop...")

    for i, tile in enumerate(tiles):
        # 1. Get the geometry for this specific tile
        roi_geometry = ee.Geometry(tile['geometry'])
        
        # 2. Define filenames
        # We pad the numbers (001, 002) so they sort correctly
        img_name = f"s2_{i:04d}.tif"
        mask_name = f"mask_{i:04d}.tif"
        
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, mask_name)

        # 3. Check if file exists (Resume capability)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            print(f"[{i+1}/{total_tiles}] Skipping (Already exists)")
            continue

        print(f"[{i+1}/{total_tiles}] Downloading...")

        try:
            # 4. Download Image
            # We use the singular 'download_ee_image' which avoids the bug
            geemap.download_ee_image(
                image=s2,
                filename=img_path,
                region=roi_geometry,
                crs='EPSG:4326',
                scale=10,
                overwrite=True
            )

            # 5. Download Mask
            geemap.download_ee_image(
                image=forest_mask,
                filename=mask_path,
                region=roi_geometry,
                crs='EPSG:4326',
                scale=10,
                overwrite=True
            )
        except Exception as e:
            print(f"Error downloading tile {i}: {e}")

    print("Download Complete!")

if __name__ == "__main__":
    main()