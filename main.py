import argparse
import pandas as pd
import logging
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from utils import prepare_dataset
from evaluation.tasks.fairness_learning import FairnessBenchmark
from algorithms.fairness_learning.lfr import LFR
from algorithms.fairness_learning.grid_search_reduction import GridSearchReduction
from metrics.binary_fairness_metrics import BinaryLabelFairnessMetric

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--task", type=str, required=True, choices=["fair", "oodg", "ood", "fairdg"], help="Type of task, fair(fairness learning), oodg (OOD generalization), oodd (OOD detection), fairdg (fariness-aware domain generalization)")
    parser.add_argument("--dataset", type=str, required=True, choices=["f4d", "celeba", "fairface", "utkface", "utk-fairface"], help="Path to the dataset CSV file")
    parser.add_argument("--label", type=str, required=True, help="Name of the label column")
    
    
    args, unknown = parser.parse_known_args()
    
    if args.task == "fair":
        parser.add_argument("--sensitive", type=str, required=True, help="Name of the sensitive attribute column")
        parser.add_argument("--domain", type=str, default='', help="Attributie for domain division")
        parser.add_argument("--model", type=str, required=True, help="['lfr','']")
    elif args.task == "dg":
        parser.add_argument("--domain", type=str, required=True, help="Attribute for domain division")
        parser.add_argument("--sensitive", type=str, required=True, help="Name of the sensitive attribute column")

    
    args = parser.parse_args()

    # logging
    logging.info("Running with the following configuration:")
    args_table = [[arg, value] for arg, value in vars(args).items()]
    logging.info("\n" + tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    
    # Load data and run the benchmark
    if args.task == "fair":
        dataset = prepare_dataset(args.dataset, args.task, label = args.label, sensitive = args.sensitive)

        # Configure model
        if args.model == "lfr":
            model = LFR(k=10, Ax=0.1, Ay=1.0, Az=2.0,verbose=1)
            model = model.fit(dataset.train_dataset[0], maxiter=5000, maxfun=5000)
            
            dataset_transf_test = model.transform(dataset.test_dataset[0])

            metrics = BinaryLabelFairnessMetric(dataset.test_dataset[0].labels, dataset_transf_test.labels, dataset.test_dataset[0].sensitive_attribute)
            
            # metric_transf_test = BinaryLabelFairnessMetric(dataset_transf_test)
            # print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_test.mean_difference())
        elif args.model == 'grid_search_reduction':
            estimator = LogisticRegression(solver='liblinear', random_state=1234)
            model = GridSearchReduction(estimator=estimator, 
                                    #   constraints="EqualizedOdds",
                                      constraints="DemographicParity",
                                      grid_size=20)
            model.fit(dataset.train_dataset[0])
            dataset_transf_test = model.predict(dataset.test_dataset[0])

            metrics = BinaryLabelFairnessMetric(dataset.test_dataset[0].labels, dataset_transf_test.labels, dataset.test_dataset[0].sensitive_attribute)
        else:
            raise ValueError("Unsupported model type")

        # benchmark = FairnessBenchmark(dataset, args.label, args.sensitive, model)
        # benchmark = FairnessBenchmark()
        # results = benchmark.run()
    else:
        logging.info("Domain Generalization task is not yet implemented.")
        return
    
    logging.info("Final Evaluation Results:")
    results = [['Accuracy',metrics.accuracy()], ['Difference_DP',metrics.difference_DP()], ['Difference_EO',metrics.difference_EO()]]
    logging.info("\n" + tabulate(results, headers=["Metric", "Result"], tablefmt="grid"))

if __name__ == "__main__":
    main()
