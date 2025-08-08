import tensorflow as tf
from tensorflow import keras
import psutil

import os
import numpy as np
import pandas as pd
import pickle
import dotenv
import boto3
import matplotlib.pyplot as plt
import matplotlib.patches as patches


for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
ram_gb = psutil.virtual_memory().total / 1e9

print("TF version:", tf.__version__) # need 2.18
print("GPUs:", tf.config.list_physical_devices("GPU"))
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))


labels_df = pd.read_csv(
    'C:/Users/ANNIE CHEN/Box Sync/Personal/flags/flags_annotations_df.csv'
)