#%%
import os
import numpy as np
import pandas as pd
import pickle
import dotenv
import boto3
from botocore.config import Config
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from io import BytesIO # create file-like object from bytes
from PIL import Image, UnidentifiedImageError # image data

from utils import (
    setup_s3_client,
    read_from_s3,
    parse_xml
)

# Path to images and annotations
# Set your AWS credentials as environment variables
dotenv.load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_client = setup_s3_client(access_key, secret_key, max_connections=100)
bucket_name = "flags-estimation"

#%%
class_ids = [
    "flags"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# grab from bucket and parse xml into df
annotations_xml = read_from_s3(s3_client = s3_client,
                               bucket_name = bucket_name,
                               object_key = "data/mapillary/flags_annotations_v4.xml",
                               object_type = "XML")
labels_df = parse_xml(annotations_xml)
#labels_df = labels_df.set_index("id")
labels_df['image_index'] = pd.factorize(labels_df['filename'])[0] + 1
labels_df = labels_df[labels_df.image_index.isin(range(10500))]


#%%
# write out
labels_df.to_csv("C:/Users/ANNIE CHEN/Box Sync/Personal/flags/flags_annotations_df.csv", index=False)
#%%