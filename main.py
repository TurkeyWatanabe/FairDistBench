import argparse
import pandas as pd
import logging
import os
from datetime import datetime
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import torch

from utils import prepare_dataset
from evaluation.tasks.fairness_learning import FairnessBenchmark
from algorithms.fairness_learning.lfr import LFR
from algorithms.fairness_learning.gsr import GridSearchReduction
from algorithms.fairness_learning.ad import AdversarialDebiasing
from algorithms.domain_generalization.erm import ERM
from algorithms.domain_generalization.irm import IRM
from algorithms.domain_generalization.gdro import GroupDRO
from algorithms.domain_generalization.mixup import Mixup
from algorithms.domain_generalization.mmd import MMD
from algorithms.domain_generalization.mbdg import MBDG

from metrics.binary_fairness_metrics import BinaryLabelFairnessMetric
from metrics.domain_generalization_metrics import DomainGeneralizationMetric

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

current_directory = os.getcwd()
log_file_path = os.path.join(current_directory, "logfile.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(
    filename=log_file_path,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--task", type=str, required=True, choices=["fair", "oodg", "oodd-s", "oodd-a", "oodd-e", "fairdg"], help="Type of task, fair(fairness learning), oodg (OOD generalization), oodd (OOD detection, oodd-s(sensory), oodd-a(intra-domain semantic), oodd-e(inter-domain semantic)), fairdg (fariness-aware domain generalization)")
    parser.add_argument("--dataset", type=str, required=True, choices=["f4d", "celeba", "fairface", "utkface", "utk-fairface"], help="Path to the dataset CSV file")
    parser.add_argument("--label", type=str, required=True, help="Name of the label column")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=1, help="Epoch for training")
    parser.add_argument("--n_steps", type=int, default=1, help="Steps in each epoch")
    
    
    args, unknown = parser.parse_known_args()
    
    if args.task == "fair":
        parser.add_argument("--sensitive", type=str, required=True, help="Name of the sensitive attribute column")
        parser.add_argument("--domain", type=str, default='', help="No need for fair task")
        parser.add_argument("--model", type=str, required=True, choices=["lfr", "gsr","ad"])
    elif args.task == "oodg":
        parser.add_argument("--domain", type=str, required=True, help="Attribute for domain division")
        parser.add_argument("--sensitive", type=str, default='', help="No need for oodg task")
        parser.add_argument("--model", type=str, required=True, choices=["erm", "irm","gdro","mixup","mmd","mbdg"])
    elif args.task == "oodd-s":
        parser.add_argument("--domain", type=str, required=True, help="Attribute for domain division")
        parser.add_argument("--sensitive", type=str, default='', help="No need for oodg task")
        parser.add_argument("--model", type=str, required=True, choices=["svm", "ddu","msp","energy","entropy"])
    

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
            estimator = LogisticRegression(solver='liblinear')
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

        accs = []
        f1s = []
        for i in range(dataset.num_domains):
            logging.info(f"Leave domian {i} for testing...")
            if args.model == 'erm':
                model = ERM(batch_size=args.batch_size, epoch=args.epoch, n_steps=args.n_steps)
            elif args.model == 'irm':
                model = IRM(batch_size=args.batch_size, epoch=args.epoch, n_steps=args.n_steps)
            elif args.model == 'gdro':
                model = GroupDRO(batch_size=args.batch_size, epoch=args.epoch, n_steps=args.n_steps)
            elif args.model == 'mixup':
                model = Mixup(batch_size=args.batch_size, epoch=args.epoch, n_steps=args.n_steps)
            elif args.model == 'mmd':
                model = MMD(batch_size=args.batch_size, epoch=args.epoch, n_steps=args.n_steps)
            elif args.model == 'mbdg':
                model = MBDG(batch_size=args.batch_size, epoch=args.epoch, n_steps=args.n_steps)
            else:
                raise ValueError(f"Unsupported model type for {args.task} task")
            
            model.fit(dataset.train_dataset[i])

            preds, labels = model.predict(dataset.test_dataset[i])

            metrics = DomainGeneralizationMetric(labels, preds)
            
            accs.append(metrics.accuracy())
            f1s.append(metrics.f1())
        
        
        results = [['Accuracy',sum(accs) / len(accs)], ['F1-Score',sum(f1s) / len(f1s)]]

    elif args.task =='oodd-s':
        args.sensitive = ''
        # data loader
        dataset = prepare_dataset(args.dataset, args.task, label = args.label, sensitive = args.sensitive, domain=args.domain)
        accs = []
        f1s = []
        for i in range(dataset.num_domains):
            logging.info(f"Leave domian {i} for testing...")
            print(dataset.train_dataset[i].domain)
            print(dataset.train_dataset[i].labels)

            print(dataset.test_dataset[i].domain)
            print(dataset.test_dataset[i].labels)
            print(dataset.test_dataset[i].ood_labels)
            print()

        exit(0)


    
    
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
