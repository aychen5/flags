#%% 
# import the necessary packages
import numpy as np
import pandas as pd
import geopandas as gpd
import boto3
from io import BytesIO
import logging
import argparse
import requests
import json
import mapillary.interface as mly
        
#%% 
# Get complete list of images within a bounding box
def get_boundingbox_data(bb, bb_id, path):
    print(f'Finding images in bounding box {bb_id}')
    data = json.loads(
        mly.images_in_bbox(
            bbox=bb,
            max_captured_at="*",
            min_captured_at="2005-03-15",
            image_type="all",
            compass_angle=(0, 360),
        )
    )
    # save 
    with open(f"{path}/images_in_bbox_{bb_id}.json", mode="w") as f:
        json.dump(data, f, indent=4)
    return(data)

def init_aws_session():
    session = boto3.session.Session()# make sure you have aws configured
    s3_client = session.client('s3')
    return s3_client

def upload_to_s3(s3_client,
                 request_url, 
                 key):
    ''' 
    Take image URL and upload image to S3 bucket. 
    '''
    r = requests.get(request_url)
    if r.status_code == 200:
        #convert content to bytes, since upload_fileobj requires file like obj
        bytesIO = BytesIO(bytes(r.content))    
        with bytesIO as data:
            s3_client.upload_fileobj(Fileobj=data, 
                                     Bucket="flags-estimation", 
                                     Key=f'data/mapillary/train/mapillary-{key}.jpg')
        logging.info(
            "Object '%s' to bucket '%s'" % (f"mapillary-{key}.jpg", "flags-estimation")
        )
    else:
        logging.debug(
            "Couldn't put object '%s' to bucket '%s'" % (f"mapillary-{key}.jpg", "flags-estimation")
         )

def get_image_urls(data, restart_index):
    '''
    Obtain all features in a subbox.
    Save the metadata for each feature.
    Get the image urls and save the jpg to DB.
    '''
    s3_client = init_aws_session() 
    for j in range(restart_index, len(data)):
        d = data[j]
        d_type = d['type']
        n_features = len(d['features'])
        logging.info(f'{j}: This is a {d_type} with {n_features} features.')
        metadata_results = pd.DataFrame()
        # for each feature, extra
        for i in range(n_features):
            feat = d['features'][i]
            feat_properties = feat['properties']
            # save metadata: capture timestamp, geo coordinates, camera angle
            img_metadata = pd.DataFrame([feat_properties])
            img_metadata['longitude'] = feat['geometry']['coordinates'][0]
            img_metadata['latitude'] = feat['geometry']['coordinates'][1]
            img_metadata['nonempty_bb_id'] = j
            img_metadata['feat_id'] = i
            metadata_results = pd.concat([metadata_results, img_metadata])
            img_metadata.to_csv(f"data/mapillary/train-999996/img-metadata-{i}.csv")

            # download image
            img_id = feat_properties['id']
            img_url = mly.image_thumbnail(
                image_id=img_id, resolution=2048
            )
            try:
                upload_to_s3(s3_client = s3_client,
                             request_url = img_url,
                             key = img_id)
                logging.info("Uploading...data[%d]['features'][%d]['properties']['id']" % (j, i) )
            except:
                pass
        # save
        metadata_results.to_csv(f"data/mapillary/train-999996/mapillary-metadata-{j}.csv")

#%%    
if __name__ == '__main__':
    # Mapillary access token -- provide your own, replace this example
    mly_key = 'MLY|10059641264106038|df465b736faf3da4b67e7e2ebf0454eb'
    mly.set_access_token(mly_key)

    # set up logger
    logging.basicConfig(
        filename="logs/image_download_logger.log",
        filemode='a', # append
        format="%(levelname)s | %(asctime)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.INFO # minimum level accepteds
    )

    # bounding box of NYC (roughly 100km x 50km)
    nycbb = {
        'north': 40.934743,
        'west': -74.263045,
        'south': 40.487652,
        'east': -73.675963
        }
    # divide bounding box in smaller boxes
    southwest = (nycbb['south'], nycbb['west'])
    southeast = (nycbb['south'], nycbb['east'])
    northwest = (nycbb['north'], nycbb['west'])
    northeast = (nycbb['north'], nycbb['east'])
    # say, 1km x 1km
    longitudes = np.linspace(southwest[1], southeast[1], num=101) # array is west -> east
    latitudes = np.linspace(northwest[0], southwest[0], num=51) # array is north -> south
    # each row is a box, columns with x,y coordinates for each cardinal direction
    grid = pd.DataFrame(  )
    for i in range(len(latitudes)-1):
        for j in range(len(longitudes)-1):
            print([i, j])
            l = {"north" : latitudes[i],
                "west" : longitudes[j],
                "south" : latitudes[i+1],
                "east" : longitudes[j+1]}
            grid = pd.concat([grid, pd.DataFrame([l])])
    dim = (len(latitudes)-1) * (len(longitudes)-1)
    grid["bb_id"] = range(0, dim)
    grid.reset_index(drop=True, inplace=True)

    #%% 
    OUTPUT_PATH = "C:/Users/ANNIE CHEN/Box Sync/Personal/flags/metadata/"
    # stopped at 500
    #test_bb_data = [get_boundingbox_data(bb = grid.iloc[_].to_dict(), bb_id = _, path = OUTPUT_PATH) for _ in range(500, grid.shape[0]-1)]

    more_bb_data = []
    for bb_id in range(0, grid.shape[0]-1):
        with open(f"XX/images_in_bbox_{bb_id}.json") as f:   
                d = json.load(f)
                more_bb_data.append(d)
    # take only the bounding boxes that are non-empty
    nonempty_bb_ids = [len(more_bb_data[_]['features']) != 0 for _ in range(len(more_bb_data))]
    nonempty_bb_index = np.where(nonempty_bb_ids)[0]
    nonempty_bbs = [more_bb_data[nonempty_bb_index[i]] for i in range(len(nonempty_bb_index)) ]

    #nonempty_bbs[999]['features'][22]['properties']['id']
    # restart running at 999
    get_image_urls(data = nonempty_bbs, restart_index = 999)

    #%%


#%%
# test_results = nonempty_bbs_test[1]
# test_results['bb_id'] = 3009
# #test_results.to_csv("data/mapillary/bb3009_test_metadata.csv", index=False)
#%% 
# creating the training dataset
# sc_bb = {
#     "north" : 40.761651,				
#     "west" : -73.896251,
#     "south" : 40.749818,
#     "east" : -73.880544
# }
#FLATIRON: -73.994177,40.736880,-73.981130,40.745789 (bb_id = 999999)
#HARLEM: -73.957619,40.799461,-73.944230,40.808232 (bb_id = 999998)
#DUMBO = -73.996237,40.687472,-73.983577,40.696583 (bb_id = 999997)
#JACKSON HEIGHTS = -73.896251,40.749818,-73.880544,40.761651 (bb_id = 999996)
#jackson_heights = get_boundingbox_data(bb = sc_bb, bb_id = 999996)

#get_image_urls(data = [jackson_heights], restart_index = 0)
#%% 