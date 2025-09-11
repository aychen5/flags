#%% 
# import the necessary packages
from sre_parse import FLAGS
import numpy as np
import pandas as pd
import geopandas as gpd
import boto3

import tqdm
from io import BytesIO
import logging
import dotenv
import os
import argparse
import requests
import json
import mapillary.interface as mly


#%% 
# Get complete list of images within a bounding box
def get_boundingbox_data(bb, bb_id, path):
    print(f'Finding images in bounding box {bb_id}')
    #try:
    data = mly.images_in_bbox(
            bbox=bb,
            max_captured_at=None,
            min_captured_at="2020-01-31",
            image_type="all",
            compass_angle=(0, 360),
        )
    # save 
    # with open(f"{path}/images_in_grid_{bb_id}_202025.json", mode="w") as f:
    #     json.dump(data, f, indent=4)
    # except Exception as e:
    #     logging.error(f"Error retrieving data for bounding box {bb_id}: {e}")
    #     return {"type": "FeatureCollection", "features": []}
    return(data)

#%%
# ---- config ----
FLAGS_PATH = "C:/Users/ANNIE CHEN/Box Sync/Personal/flags/"
dotenv.load_dotenv(".env")
# set up logger
logging.basicConfig(
    filename="image_ids_logger.log",
    filemode='a', # append
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    level=logging.INFO # minimum level accepteds
)

# Mapillary access token -- provide your own, replace this example
mly.set_access_token(os.getenv("MLY_KEY"))
grid = pd.read_csv(f"{FLAGS_PATH}metadata/nyc_boundingbox_grid.csv")
#%%
# --- save all metadata for each bounding box
nonempty_grid_metadata = []
for grid_id in range(2759, grid.shape[0]-1):
    print(f"Processing grid cell {grid_id}/{grid.shape[0]-1}")
    # get the metadata for each grid cell
    grid_metadata = get_boundingbox_data(bb = grid.iloc[grid_id].to_dict(),
                                         bb_id = grid_id,
                                         path = FLAGS_PATH + "metadata/images_in_grid_202025")
    # if the grid is empty, skip it
    if len(grid_metadata['features']) == 0:
        logging.info(f"Skipping empty grid cell {grid_id}")
        continue
    else:
        nonempty_grid_metadata.append(grid_metadata)
        logging.info(f"Appending grid cell {grid_id} with {len(grid_metadata['features'])} features.")

#%%
all_grid_metadata = []
for grid_id in range(0, 2760):
    print(f"Processing grid cell {grid_id}/{grid.shape[0]-1}")
    with open(f"{FLAGS_PATH}metadata/images_in_grid_202025/images_in_grid_{grid_id}_202025.json", "r") as f:
        grid_metadata = json.load(f)
    all_grid_metadata.append(grid_metadata)
#%%
# images_in_grid_202025.json has all metadata for  grid cells in a single file (empty and not empty)
with open(f"{FLAGS_PATH}metadata/images_in_grid_202025/images_in_grid_202025.json", "w") as f:
    json.dump(all_grid_metadata, f, indent=4)

nonempty_grid_metadata = [fc for fc in all_grid_metadata if fc.get("features")]  # truthy if not []
with open(f"{FLAGS_PATH}metadata/images_in_grid_202025/nonempty_grid_metadata_202025.json", "w") as f:
    json.dump(nonempty_grid_metadata, f, indent=4)
#%% 