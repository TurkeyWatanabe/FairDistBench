import argparse
import pandas as pd
import logging
import os
from datetime import datetime
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from utils import prepare_dataset
from evaluation.tasks.fairness_learning import FairnessBenchmark
from algorithms.fairness_learning.lfr import LFR
from algorithms.fairness_learning.gsr import GridSearchReduction
from algorithms.fairness_learning.ad import AdversarialDebiasing
from metrics.binary_fairness_metrics import BinaryLabelFairnessMetric

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

current_directory = os.getcwd()
log_file_path = os.path.join(current_directory, "logfile.log")


logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--task", type=str, required=True, choices=["fair", "oodg", "ood", "fairdg"], help="Type of task, fair(fairness learning), oodg (OOD generalization), oodd (OOD detection), fairdg (fariness-aware domain generalization)")
    parser.add_argument("--dataset", type=str, required=True, choices=["f4d", "celeba", "fairface", "utkface", "utk-fairface"], help="Path to the dataset CSV file")
    parser.add_argument("--label", type=str, required=True, help="Name of the label column")
    
    
    args, unknown = parser.parse_known_args()
    
    if args.task == "fair":
        parser.add_argument("--sensitive", type=str, required=True, help="Name of the sensitive attribute column")
        parser.add_argument("--domain", type=str, default='', help="No need for fair task")
        parser.add_argument("--model", type=str, required=True, choices=["lfr", "gsr","ad"])
    elif args.task == "oodg":
        parser.add_argument("--domain", type=str, required=True, help="Attribute for domain division")
        parser.add_argument("--sensitive", type=str, default='', help="No need for oodg task")
        parser.add_argument("--model", type=str, required=True, choices=["erm", "irm","gdro","ddg","mbdg"])
    

    args = parser.parse_args()
    output_file_path = os.path.join(current_directory, f"output/output_{args.task}_{args.dataset}_{args.model}_{args.label}_{args.sensitive}_{args.domain}.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # logging
    now = datetime.now()
    logging.info(now.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Running with the following configuration:")
    args_table = [[arg, value] for arg, value in vars(args).items()]
    logging.info("\n" + tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    with open(output_file_path, "a") as file:
        file.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        file.write("\nRunning with the following configuration:")
        file.write("\n" + tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    
    

    # Load data and run the benchmark
    if args.task == "fair":
        args.domain = ''
        # data loader
        dataset = prepare_dataset(args.dataset, args.task, label = args.label, sensitive = args.sensitive, domain=args.domain)

        # Configure model
        if args.model == "lfr":
            model = LFR(k=10, Ax=0.1, Ay=1.0, Az=2.0,verbose=1)
            model = model.fit(dataset.train_dataset[0], maxiter=5000, maxfun=5000)
            
            dataset_transf_test = model.transform(dataset.test_dataset[0])

            metrics = BinaryLabelFairnessMetric(dataset.test_dataset[0].labels, dataset_transf_test.labels, dataset.test_dataset[0].sensitive_attribute)
            
            # metric_transf_test = BinaryLabelFairnessMetric(dataset_transf_test)
            # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_test.mean_difference())
        elif args.model == 'gsr':
            estimator = LogisticRegression(solver='liblinear', random_state=1234)
            model = GridSearchReduction(estimator=estimator, 
                                    #   constraints="EqualizedOdds",
                                      constraints="DemographicParity",
                                      grid_size=20)
            model.fit(dataset.train_dataset[0])
            dataset_transf_test = model.predict(dataset.test_dataset[0])

            metrics = BinaryLabelFairnessMetric(dataset.test_dataset[0].labels, dataset_transf_test.labels, dataset.test_dataset[0].sensitive_attribute)
        
        elif args.model == 'ad':
            sess = tf.Session()
            model = AdversarialDebiasing(
                          scope_name='plain_classifier',
                          debias=False,
                          sess=sess)
            model.fit(dataset.test_dataset[0])

            dataset_transf_test = model.predict(dataset.test_dataset[0])

            metrics = BinaryLabelFairnessMetric(dataset.test_dataset[0].labels, dataset_transf_test.labels, dataset.test_dataset[0].sensitive_attribute)

        else:
            raise ValueError(f"Unsupported model type for {args.task} task")
        
        results = [['Accuracy',metrics.accuracy()], ['Difference_DP',metrics.difference_DP()], ['Difference_EO',metrics.difference_EO()]]

    elif args.task == "oodg":
        args.sensitive = ''
        # data loader
        dataset = prepare_dataset(args.dataset, args.task, label = args.label, sensitive = args.sensitive, domain=args.domain)

        if args.model == 'erm':
            for i in range(dataset.num_domains):
                pass
                # print(len(dataset.train_dataset[i]))
                # print(len(dataset.test_dataset[i]))
        else:
            raise ValueError(f"Unsupported model type for {args.task} task")

    
    
    logging.info("Final Evaluation Results:")
    logging.info("\n" + tabulate(results, headers=["Metric", "Result"], tablefmt="grid"))
    with open(output_file_path, "a") as file:
        file.write("\nFinal Evaluation Results:")
        file.write("\n" + tabulate(results, headers=["Metric", "Result"], tablefmt="grid"))
        file.write("\n\n")

if __name__ == "__main__":
    try:
        main()
    finally:
        logging.shutdown()
