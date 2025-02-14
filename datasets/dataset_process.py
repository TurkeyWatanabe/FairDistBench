import os
import json
import logging
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, data, labels, sensitive_attribute, domain):
        self.data = data  # The main data (e.g., features of the dataset)
        self.labels = labels  # The labels (e.g., target values for classification)
        self.sensitive_attribute = sensitive_attribute  # Sensitive attribute (e.g., gender, race)
        self.domain = domain  # Domain of the dataset (e.g., P, A, C, S)

        self.train_dataset = []
        self.test_dataset = []

    def __len__(self):
        return len(self.labels)
    
    def devide_by_domain(self):
        datasets = []
        if len(self.domain) != 0:
            pass 
        else:
            datasets.append(Dataset(
                data=self.data,
                labels=self.labels,
                sensitive_attribute=self.sensitive_attribute,
                domain=self.domain
            ))

        self.devided_dataset = datasets
            
    
    def split_dataset(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets by domains.

        Args:
            dataset (Dataset): Dataset from one domain.
            test_size (float): The proportion of the dataset to be used as the test set.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: (train_dataset, test_dataset)
        """
        # Split the indices for training and testing
        for dataset in self.devided_dataset:
            indices = np.arange(len(dataset))
            train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

            train_dataset = Dataset(
                data=dataset.data[train_indices],
                labels=dataset.labels[train_indices],
                sensitive_attribute=dataset.sensitive_attribute[train_indices] if len(dataset.sensitive_attribute) != 0 else [],
                domain=dataset.domain[train_indices] if len(dataset.domain) != 0 else []
            )

            test_dataset = Dataset(
                data=dataset.data[test_indices],
                labels=dataset.labels[test_indices],
                sensitive_attribute=dataset.sensitive_attribute[test_indices]if len(dataset.sensitive_attribute) != 0 else [],
                domain=dataset.domain[test_indices] if len(dataset.domain) != 0 else []
            )
            
            self.train_dataset.append(train_dataset)
            self.test_dataset.append(test_dataset)
            

    def __repr__(self):
        return f"Dataset(data={self.data.shape}, labels={self.labels.shape}, sensitive_attribute={self.sensitive_attribute.shape}, domain={self.domain})"

def load_data(dataset, task, path, label, sensitive, domain):

    #### data loader
    
    resized_dir = os.path.join(path, 'resized')
    anno_file = os.path.join(path, 'anno', dataset+'.json')
    
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

        # Construct the image filename, ensuring it has leading zeros (e.g., 000001.jpg)
        image_filename = f"{image_id:06d}.jpg" # Format id as a 5-digit number (e.g., 000001.jpg)
        image_path = os.path.join(resized_dir, image_filename)
        
        # Load the image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        images.append(img_array)
        if label == 'age':
            age = get_age(task, dataset, item)
            labels.append(age)
        elif label == 'gender':
            gender = get_gender(dataset, item)
            labels.append(gender)
        elif label != '':
            try:
                labels.append(item[label])
            except KeyError as e:
                logging.error(f"Missing key for {dataset}: {e}")
                raise KeyError(f"Missing key for {dataset}: {e}")
            
        if sensitive == 'age':
            age = get_age(task, dataset, item)
            sensitive_attributes.append(age)
        elif sensitive == 'gender':
            gender = get_gender(dataset, item)
            sensitive_attributes.append(gender)
        elif sensitive != '':
            try:
                sensitive_attributes.append(item[sensitive]) 
            except KeyError as e:
                logging.error(f"Missing key for {dataset}: {e}")
                raise KeyError(f"Missing key for {dataset}: {e}")
            
        if domain != '':
            try:
                domains.append(item[domain])
            except KeyError as e:
                logging.error(f"Missing key for {dataset}: {e}")
                raise KeyError(f"Missing key for {dataset}: {e}")
             

        cnt += 1
        if cnt % 10000 == 0 and cnt != 0:
            logging.info(f"{cnt} images from {dataset} have been loaded")

    images = np.array(images)
    labels = np.array(labels)
    sensitive_attributes = np.array(sensitive_attributes)
    domains = np.array(domains)

    dataset = Dataset(data=images, labels=labels, sensitive_attribute=sensitive_attributes, domain=domains)


    ### domain division
    dataset.devide_by_domain()

    ### dataset split
    dataset.split_dataset()


    return dataset


def get_gender(dataset, item):
    '''
    Specific gender process for different datasets.

    return: 0: Male; 1: Female

    '''
    if dataset == 'f4d':
        try:
            gender = item['gender']
        except KeyError:
            raise KeyError(f"Missing 'age' key for id {item['id']}")

        return gender - 1
    elif dataset == 'celeba':
        try:
            male = item['Male']
        except KeyError:
            raise KeyError(f"Missing 'age' key for id {item['id']}")
        
        if male == 1: return 0
        elif male == -1: return 1
    elif dataset == 'fairface':
        try:
            gender = item['gender']
        except KeyError:
            raise KeyError(f"Missing 'age' key for id {item['id']}")
        
        if gender == 'Male': return 0
        elif gender == "Female": return 1
    elif dataset == 'utkface':
        try:
            gender = item['gender']
        except KeyError:
            raise KeyError(f"Missing 'age' key for id {item['id']}")

        return gender
    elif dataset =='utk-fairface':
        pass


def get_age(task, dataset, item):
    '''
    Specific age process for different tasks and different datasets.

    '''

    if task[0:4]=='oodd':
        if dataset == 'f4d':
            try:
                teenager = item['teenager']
            except KeyError:
                raise KeyError(f"Missing 'teenager' key for id {item['id']}")
            try:
                middle = item['middle']
            except KeyError:
                raise KeyError(f"Missing 'middle' key for id {item['id']}")
            try:
                elderly = item['elderly']
            except KeyError:
                raise KeyError(f"Missing 'elderly' key for id {item['id']}")

            if teenager == 1:
                return 0  # Teenager
            elif middle == 1:
                return 1  # Middle-aged
            elif elderly == 1:
                return 2  # Elderly
            else:
                # If no valid age category is found, raise an exception
                raise ValueError(f"Invalid age category data: {item}")

        elif dataset == 'celeba':
            pass
        elif dataset == 'fairface':
            try:
                age = item['age']
            except KeyError:
                raise KeyError(f"Missing 'age' key for id {item['id']}")
            if age in ["0-2", "3-9", "10-19", "20-29"]:  # Category 0
                return 0
            elif age in ["30-39", "40-49", "50-59"]:  # Category 1
                return 1
            elif age in ["60-69", "more than 70"]:  # Category 2
                return 2
            else:
                raise ValueError(f"Unknown age group: {age} for id {item['id']}")
        elif dataset == 'utkface':
            try:
                age = int(item['age'])
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid or missing 'age' value for id {item['id']}: {e}")
            if 0 <= age <= 29:
                return 0
            elif 30 <= age <= 59:
                return 1
            elif 60 <= age <= 116:
                return 2
            else:
                # If age is outside the expected range (0-116), raise an exception
                raise ValueError(f"Age value out of expected range (0-116) for id {item['id']}: {age}")
        elif dataset =='utk-fairface':
            try:
                age = item['age']
            except KeyError:
                raise KeyError(f"Missing 'age' key for id {item['id']}")
            if age in ["0-2", "3-9", "10-19", "20-29"]:  # Category 0
                return 0
            elif age in ["30-39", "40-49", "50-59"]:  # Category 1
                return 1
            elif age in ["60-69", "more than 70"]:  # Category 2
                return 2
            else:
                raise ValueError(f"Unknown age group: {age} for id {item['id']}")

    else:
        if dataset == 'f4d':
            try:
                teenager = item['teenager']
            except KeyError:
                raise KeyError(f"Missing 'teenager' key for id {item['id']}")
            try:
                middle = item['middle']
            except KeyError:
                raise KeyError(f"Missing 'middle' key for id {item['id']}")
            try:
                elderly = item['elderly']
            except KeyError:
                raise KeyError(f"Missing 'elderly' key for id {item['id']}")
            if teenager == 1 or middle == 1:
                return 0
            elif elderly == 1:
                return 1
            else:
                # If no valid age category found, raise an exception
                raise ValueError(f"Invalid age category data: {item}")
        elif dataset == 'celeba':
            try:
                young = item['Young']
            except KeyError:
                raise KeyError(f"Missing 'Young' key for id {item['id']}")

            if young == 1:
                return 0
            else:
                return 1

        elif dataset == 'fairface':
            try:
                age = item['age']
            except KeyError:
                raise KeyError(f"Missing 'age' key for id {item['id']}")
            if age in ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49"]:
                return 0
            elif age in ["50-59", "60-69", "more than 70"]:
                return 1
            else:
                raise ValueError(f"Unknown age group: {age} for id {item['id']}")
        elif dataset == 'utkface':
            try:
                age = int(item['age'])
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid or missing 'age' value for id {item['id']}: {e}")
            if age >= 50:
                return 1
            elif age < 50:
                return 0
            else:
                # If age is outside the expected range (0-116), raise an exception
                raise ValueError(f"Age value out of expected range (0-116) for id {item['id']}: {age}")
        elif dataset =='utk-fairface':
            try:
                age = item['age']
            except KeyError:
                raise KeyError(f"Missing 'age' key for id {item['id']}")
            if age in ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49"]:
                return 0
            elif age in ["50-59", "60-69", "more than 70"]:
                return 1
            else:
                raise ValueError(f"Unknown age group: {age} for id {item['id']}")