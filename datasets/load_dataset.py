import os
import json
import logging
import numpy as np
from PIL import Image

class Dataset:

    def __init__(self, data, labels, sensitive_attribute, domain):
        self.data = data  # The main data (e.g., features of the dataset)
        self.labels = labels  # The labels (e.g., target values for classification)
        self.sensitive_attribute = sensitive_attribute  # Sensitive attribute (e.g., gender, race)
        self.domain = domain  # Domain of the dataset (e.g., P, A, C, S)

    def __repr__(self):
        return f"Dataset(data={self.data.shape}, labels={self.labels.shape}, sensitive_attribute={self.sensitive_attribute.shape}, domain={self.domain})"

def load_fairface(path, label, sensitive, domain):
    
    resized_dir = os.path.join(path, 'resized')
    anno_file = os.path.join(path, 'anno', 'fairface.json')
    
    with open(anno_file, 'r') as f:
        annotations = json.load(f)

    images = []
    labels = []
    sensitive_attributes = []
    domains = []

    # Process each item in the annotations
    cnt = 0
    for item in annotations:
        image_id = item['id']

        # Construct the image filename, ensuring it has leading zeros (e.g., 00001.jpg)
        image_filename = f"{image_id:05d}.jpg"  # Format id as a 5-digit number (e.g., 00001.jpg)
        image_path = os.path.join(resized_dir, image_filename)
        
        # Load the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        images.append(img_array)
        if label != '':
            labels.append(item[label])
        if sensitive != '':
            sensitive_attributes.append(item[sensitive]) 
        if domain != '':
            domains.append(item[domain]) 

        cnt += 1
        if cnt % 10000 == 0 and cnt != 0:
            logging.info("{} images from Fairface have been loaded".format(cnt))

    images = np.array(images)
    labels = np.array(labels)
    sensitive_attributes = np.array(sensitive_attributes)
    domains = np.array(domains)

    dataset = Dataset(data=images, labels=labels, sensitive_attribute=sensitive_attributes, domain=domains)

    return dataset