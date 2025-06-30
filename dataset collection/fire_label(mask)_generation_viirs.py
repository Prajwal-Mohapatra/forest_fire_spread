
# ===================== SETUP =====================
!pip install geopandas rasterio fiona pyproj shapely --quiet

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from datetime import datetime
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# ===================== PATHS =====================
# Change these paths according to your Drive structure
shapefile_path = "/content/fire_archive_SV-C2_627645.shp"
reference_raster_path = "/content/drive/MyDrive/Dataset_PS1/uttarakhand_fuel_map.tif"
output_dir = "/content/drive/MyDrive/Dataset_PS1/fire_masks_april_2023"

os.makedirs(output_dir, exist_ok=True)

# ===================== LOAD DATA =====================
# Load VIIRS fire points
gdf = gpd.read_file(shapefile_path)
gdf['ACQ_DATE'] = pd.to_datetime(gdf['ACQ_DATE'])

# Filter to high/nominal confidence only (optional)
if 'CONFIDENCE' in gdf.columns:
    gdf = gdf[gdf['CONFIDENCE'].isin(['n', 'h'])]

# Load reference raster to match resolution, CRS, transform
with rasterio.open(reference_raster_path) as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_shape = (ref.height, ref.width)

# Reproject fire points to match reference CRS
if gdf.crs != ref_crs:
    gdf = gdf.to_crs(ref_crs)

# ===================== RASTERIZE LOOP =====================
start_date = "2023-04-01"
end_date = "2023-04-30"
date_range = pd.date_range(start=start_date, end=end_date)

print("Generating daily fire masks...")

for day in tqdm(date_range):
    try:
        day_points = gdf[gdf['ACQ_DATE'] == day]
        shapes = ((geom, 1) for geom in day_points.geometry)

        fire_mask = rasterize(
            shapes=shapes,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=0,
            dtype='uint8'
        )

        output_path = os.path.join(output_dir, f"fire_mask_{day.strftime('%Y_%m_%d')}.tif")

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=fire_mask.shape[0],
            width=fire_mask.shape[1],
            count=1,
            dtype='uint8',
            crs=ref_crs,
            transform=ref_transform,
        ) as dst:
            dst.write(fire_mask, 1)
    except Exception as e:
        print(f"Error processing {day.date()}: {e}")

print("All fire masks generated!")

