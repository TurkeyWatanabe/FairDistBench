from datasets.dataset_process import *
import logging
import os
import pickle

def prepare_dataset(file_path, task, label='', sensitive='', domain=''):
    """Load and process dataset."""
    path = "datasets/" + file_path
    prepared_path = os.path.join(path, 'prepared', task)
    os.makedirs(prepared_path, exist_ok=True)
    save_path = os.path.join(prepared_path, file_path+'['+label+','+sensitive+','+domain+']'+'.pkl')

    # load
    if os.path.exists(save_path):
        logging.info("Loading data from existed file...")
        with open(save_path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        logging.info(f"Loading data from {file_path}...")
        dataset = load_data(file_path, task, path, label, sensitive, domain)
        # with open(save_path, 'wb') as f:
        #     pickle.dump(raw_dataset, f)

    logging.info(f"Finish data load from {file_path}!")
    # print(raw_dataset.data[0:5])
    print(dataset.labels[0:10])
    print(dataset.sensitive_attribute[0:10])
    print(dataset.domain[0:10])
    print(dataset.devided_dataset)
    print(dataset.train_dataset)
    print(dataset.test_dataset)
    exit(0)


    return dataset

