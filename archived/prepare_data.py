#%%
# import the necessary packages
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import glob
# AWS S
import boto3
from botocore.exceptions import NoCredentialsError

import os
from collections import namedtuple
from io import BytesIO # create file-like object from bytes
from PIL import Image # image data#%%

import torch
import torchvision
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader

# download helpers for evaluation in object detection
download_dir = "flags_venv/lib/python3.10/site-packages"
evaluation_fxns = [
   "engine.py",
   "utils.py",
   "coco_utils.py",
   "coco_eval.py",
   "transforms.py"
]
# Download the file to the specified directory
[os.system(f"wget -P {download_dir} https://raw.githubusercontent.com/pytorch/vision/main/references/detection/{_}") for _ in evaluation_fxns]
from engine import evaluate  # Import the evaluate function

# === PARAMS
# get AWS credentials from environment variables (in activate script)
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_client = boto3.client('s3')
bucket_name = "flags-estimation"

# %%
# === FUNCTIONS
def list_objects_in_bucket(bucket_name):
  '''
  Get all objects given the S3 bucket name. 
  '''
  try:
      # List all objects in the specified bucket
      response = s3_client.list_objects_v2(Bucket=bucket_name)

      # Print the object keys
      if 'Contents' in response:
          for obj in response['Contents']:
              print(f"Object Key: {obj['Key']}")
      else:
          print("No objects found in the bucket.")

  except Exception as e:
      print(f"Error: {e}")

def read_from_s3(bucket_name, object_key, object_type, as_bytes=False):
  '''
  Pull the data from S3. 
  '''
  try:
      # Get the object from S3 bucket
      response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
      # Read the content as bytes
      object_content = response['Body'].read()
      print(f"Read content from {object_key} in {bucket_name}")

      # Decode bytes and parse based on object type
      # Use BytesIO to create a file-like object from bytes
      f = BytesIO(object_content)
      if object_type=="CSV":
        out = pd.read_csv(f)
      elif object_type=="XML":
        out = BeautifulSoup(f, features='lxml')
        #out =ET.parse(f)
      elif object_type == "JPG":
          try:
              if as_bytes:
                  out = object_content
              else:
                  out = Image.open(f)
          except UnidentifiedImageError:
              print(f"Error: Cannot identify image file {object_key}")
              out = None
      else:
        out = object_content
      return out
  except NoCredentialsError:
      print("Credentials not available")
  except Exception as e:
      print(f"Error: {e}")

def split_df(df, group):
  '''
  Want to group data by image, as there are images where there are more than one flag. 
  '''
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


class FlagsDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_df, bucket_name, read_from_s3, transforms=None):
        self.annotations_df = annotations_df
        self.bucket_name = bucket_name
        self.read_from_s3 = read_from_s3
        self.transforms = transforms

    def __getitem__(self, idx):
        # Get image filename from DataFrame
        img_filename = self.annotations_df.iloc[idx]['filename']

        # Load image from S3
        img = self.read_from_s3(self.bucket_name, img_filename, "JPG")

        # Extract bounding box coordinates
        xmin = self.annotations_df.iloc[idx]['xtl']
        ymin = self.annotations_df.iloc[idx]['ytl']
        xmax = self.annotations_df.iloc[idx]['xbr']
        ymax = self.annotations_df.iloc[idx]['ybr']

        # If any coordinate is NaN, there are no bounding boxes
        if pd.isna(xmin):
           # Convert NaN values to 0
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area = torch.zeros(0, dtype=torch.float32)

        else: # if there is a bounding box in the image
            # Convert coordinates to float
            xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
            boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            labels = torch.ones(1, dtype=torch.int64)  # Assuming there's only one class
            area = (xmax - xmin) * (ymax - ymin)

        target = {"boxes": boxes, "labels": labels, "area": area}

        # transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations_df)



#%%
metadata_dir = ['train-999996', 'train-999997', 'train-999998', 'train-999999']
metadata_df_list = []
for d in metadata_dir:
   dir_path = f"/Users/anniechen/Dropbox/ML/Computer Vision/GSV/data/mapillary/{d}/"
   csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
   for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        metadata_df_list.append(df)

metadata_df = pd.concat(metadata_df_list).drop_duplicates().reset_index(drop=True)
# note that some images have multiple flags!
metadata_df['filename'] = metadata_df['id'].astype(str).apply(lambda x: 'mapillary-' + x + '.jpg') 
metadata_df = metadata_df.drop(columns=['feat_id']).drop_duplicates().reset_index(drop=True)
annotations = read_from_s3(bucket_name = bucket_name,
                           object_key = "data/mapillary/flags_annotations_v2.xml",
                           object_type = "XML")
# For each image in the labeled data, pull the bounding box information if it exists
annotations_df = pd.DataFrame()

for child in annotations.find_all():
  tag = child.name
  if tag=="image":
    class_label = None
  elif tag=="box":
    class_label = "flag"
  if tag=="box":
    # get image id
    image_parent = child.find_parent()
    id = image_parent.get(key = "id")
    filename = image_parent.get(key = "name")
    # get bounding box
    xtl = child.get(key = "xtl")
    xbr = child.get(key = "xbr")
    ytl = child.get(key = "ytl")
    ybr = child.get(key = "ybr")
    result = {
       "parent_tag" : [tag],
       "class" : [class_label],
       "annotation_id" : [int(id)],
       "filename" : [filename],
       "xtl" : [xtl], # x,y coordinates for top left
       "ytl" : [ytl],
       "xbr" : [xbr], # x,y coordinates for bottom right
       "ybr" : [ybr]
    }
    annotations_df = pd.concat([annotations_df, pd.DataFrame(result) ])
  elif tag=="image":
    # only capture image if there isn't a bounding box annotation.
    if child.find('box') == None:
      id = child.get(key = "id")
      filename = child.get(key = "name")
      result = {
        "parent_tag" : [tag],
        "class" : [class_label],
        "annotation_id" : [int(id)],
        "filename" : [filename],
        "xtl" : [None],
        "xbr" : [None],
        "ytl" : [None],
        "ybr" : [None]
      }
      annotations_df = pd.concat([annotations_df, pd.DataFrame(result) ])
    else:
      pass
  else:
    pass
  
annotations_df2 = annotations_df[annotations_df.annotation_id.isin(range(6263))].reset_index(drop=True)# %%

#%%
transformations = T.Compose([
    T.Resize((300, 300)),  # Resize image to a fixed size
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # these are best values based on ImageNet dataset
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
])

#######################################################
#                  Create Dataset Splits
#######################################################
# Set the seed for reproducibility
np.random.seed(5060)

# Generate random indices within the specified range
random_train_idx = np.random.choice(np.arange(0, annotations_df2.shape[0]),
                                    size=int(annotations_df2.shape[0] * .80), # take 80% for training
                                    replace=False)

# Select rows from the DataFrame based on the random indices for training
training = annotations_df2.iloc[random_train_idx]

# For testing, select the rows not included in the training set
testing = annotations_df2.drop(random_train_idx)


# Create an instance of the dataset
training_data = FlagsDataset(training,
                             "flags-estimation",
                             read_from_s3,
                             transforms=transformations)
test_data = FlagsDataset(testing,
                         "flags-estimation",
                         read_from_s3,
                         transforms=transformations)

def collate_fn(batch):
    # Extract images and targets from the batch
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    # Return images and targets as a tuple
    return images, targets

# create batches
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    collate_fn = collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    test_data,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn = collate_fn
)


#%%
#######################################################
#                  Build Model
#######################################################
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50

# Load a pre-trained ResNet-50 model
backbone = resnet50(pretrained=True)
# Remove the fully connected layer
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
# Set the number of output channels in the backbone
backbone.out_channels = 2048

# Create an anchor generator for the FPN
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Create a RoI align layer
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'],
    output_size=7,
    sampling_ratio=2
)

# Create the Faster R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=2,  # 91 is the number of classes in COCO dataset
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler
)

# Now you can access the roi_heads attribute
print(model.roi_heads)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#%%

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD( # SGD
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 2

# Train the model
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()

    # Initialize total loss for the epoch
    total_loss = 0.0

    # Iterate over the data loader for training data
    for images, targets in data_loader:
        # Move images and targets to the appropriate device
        # images = images.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Move images and targets to the appropriate device
        images = images.to(device) if isinstance(images, torch.Tensor) else images
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)

        # Compute the total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()

        # Update the parameters
        optimizer.step()

        # Accumulate the total loss for the epoch
        total_loss += losses.item()

    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(data_loader)}")

    # Update the learning rate
    lr_scheduler.step()

    # Evaluate the model on the test dataset
    evaluate(model, data_loader_test, device=device)
#%%