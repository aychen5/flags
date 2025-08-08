#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#%%
import requests
import os
import json
import geopandas as gpd
import pandas as pd
import settings as init
#import google_streetview.api # GSV api
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from tqdm import tqdm
import time
import argparse
#%%
def upload_to_bucket(s3_resource, file_name):
    '''
    Upload file to s3 bucket.
    '''
    bucket_name = 'flags-estimation'
    s3_resource.Bucket(bucket_name).upload_file(Filename = file_name, Key = file_name)
    print(f"File {file_name} uploaded to bucket {bucket_name}")

def calculate_lateral_heading(road_subsegment):
    fwd = road_subsegment.bearings['fwd_azimuth'] # these weren't standardized
    bwd = road_subsegment.bearings['back_azimuth']
    if bwd <=  180:
        east = bwd + 90
        if bwd <= 90: #[0, 90]
            west = bwd + 270
        elif bwd > 90: #(90, 180]
            west = bwd - 90
    elif bwd > 180:
        east = bwd + 90
        if bwd <= 90: #[0, 90]
            west = bwd + 270
        elif bwd > 90: #(90, 180]
            west = bwd - 90
    return east, west

def define_params(road_subsegment, api_key, **kwargs):
    '''
    Define the params for the metadata requests (free) and image requests (paid).
    INPUT: a road subsegment (a row in the GeoDataFrame)
    '''
    coordinates = road_subsegment.midpoints
    coordinates_str = ",".join([str(coord) for coord in coordinates])
    heading_est = calculate_lateral_heading(road_subsegment)[0]
    heading_wst = calculate_lateral_heading(road_subsegment)[1]
    meta_args = {
        'key': api_key,
        'location': coordinates_str
     }
     # for each point, take two snapshots (one in each direction: +/- 90 degrees)
    img_args_est = {
        'location': coordinates_str,
        'size': '640x640',
        'heading': heading_est,
        'pitch': '0',
        'key': api_key
        }
    img_args_wst = {
        'location': coordinates_str,
        'size': '640x640',
        'heading': heading_wst,
        'pitch': '0',
        'key': api_key
    }
    img_args = [img_args_est, img_args_wst]
    return img_args, meta_args

def get_metadata(meta_url, meta_args):
    '''
    Get the metadata for the image.
    OUTPUT: a response object, which contains the metadata including a pano_id for the location.
    '''
    meta_results = requests.get(meta_url, params = meta_args)
    if meta_results.ok==False:
        print("Metadata request failure.")
    return meta_results

def get_images(img_url, img_args):
    '''
    Get the image given the params.
    '''
    try:
        img_results = requests.get(img_url, params = img_args)
    except:

        print(f"Image request failure for {img_args}")
    return img_results

# %%
if __name__ == '__main__':
    # --- setup
    # see https://console.cloud.google.com/google/maps-apis/metrics?project=computer-vision-361314
    meta_url = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
    img_url = 'https://maps.googleapis.com/maps/api/streetview?'

    gsv_key = init.setup(type='GSV')
    s3 = init.setup(type='S3')

    # initialize parser
    parser = argparse.ArgumentParser(description='GSV image downloader')
    parser.add_argument(
        "-i",
        "--subsegment_index_range", 
        help="Downloads the images associated with specified subsegment indices. Default is first 10.", 
        default="0-10" 
        )
    args = parser.parse_args()
    start_ind, stop_ind = args.subsegment_index_range.split('-')

    if args.subsegment_index_range:
        print("Downloading outputs: % s" % args.subsegment_index_range)

    # --- data
    centerfile_shp = gpd.read_file("./data/Centerline_20230109/Centerline.shp")
    # each item is a sampled road segment
    road_segments_list = []
    for i in tqdm(range(1, centerfile_shp.shape[0])): 
        with open(f"./data/clean_centerline/centerline_sampled-{i}.json") as f:
            road_dict = json.load(f)
            road_subsegment = gpd.GeoDataFrame(road_dict)
            road_segments_list.append(road_subsegment)
    road_segments_gdf = pd.concat(road_segments_list, ignore_index=True)

    # --- get images
    # 14250 road segments, 2 images per segment
    # 28500 is the max number of requests per month within $200 free credit
    for i in tqdm(range(int(start_ind), int(stop_ind) )):
        segment_index = i
        img_args, meta_args = define_params(road_segments_gdf.iloc[i], api_key=gsv_key)
        # get metadata
        meta_results = get_metadata(meta_url, meta_args)
        time.sleep(1)
        # get image
        for j in range(len(img_args)):
            img_results = get_images(img_url, img_args[j])
            # upload to s3
            #upload_to_bucket(s3, f'./data/test_{j}.jpg')
            # save image to local data folder
            with open(f'./data/pilot_data/nyc-{segment_index}-{j}.jpg', 'wb') as file:
                file.write(img_results.content)
            


