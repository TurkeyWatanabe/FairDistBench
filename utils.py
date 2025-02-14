from datasets.load_dataset import *
import logging
import os
import pickle

def load_dataset(file_path, task, label='', sensitive='', domain=''):
    """Loads dataset."""
    path = "datasets/" + file_path
    prepared_path = os.path.join(path, 'prepared', task)
    os.makedirs(prepared_path, exist_ok=True)
    save_path = os.path.join(prepared_path, file_path+'['+label+','+sensitive+','+domain+']'+'.pkl')

    # load
    if file_path == "fairface":
        if os.path.exists(save_path):
            logging.info("Loading data from existed file...")
            with open(save_path, 'rb') as f:
                raw_dataset = pickle.load(f)
        else:
            logging.info("Loading data...")
            raw_dataset = load_fairface(path, label, sensitive, domain)
            with open(save_path, 'wb') as f:
                pickle.dump(raw_dataset, f)

        # print(raw_dataset.data[0:5])
        print(raw_dataset.labels[0:5])
        print(raw_dataset.sensitive_attribute[0:5])
        print(raw_dataset.domain[0:5])
        exit(0)
    else:
        raise ValueError("Unsupported dataset type.")

    
    # prepare for specific task
    
    # domain division

    # split dataset


    return dataset
