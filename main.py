import argparse
import pandas as pd
import logging
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from evaluation.tasks.fairness_learning import FairnessBenchmark
from utils import prepare_dataset
from algorithms.fairness_learning.lfr import LFR

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation")
    parser.add_argument("--task", type=str, required=True, choices=["fair", "oodg", "ood", "fairdg"], help="Type of task, fair(fairness learning), oodg (OOD generalization), oodd (OOD detection), fairdg (fariness-aware domain generalization)")
    parser.add_argument("--dataset", type=str, required=True, choices=["f4d", "celeba", "fairface", "utkface", "utk-fairface"], help="Path to the dataset CSV file")
    parser.add_argument("--label", type=str, required=True, help="Name of the label column")
    parser.add_argument("--model", type=str, required=True, help="Model to use for evaluation")
    
    args, unknown = parser.parse_known_args()
    
    if args.task == "fair":
        parser.add_argument("--sensitive", type=str, required=True, help="Name of the sensitive attribute column")
        parser.add_argument("--domain", type=str, default='', help="Attributie for domain division")
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
            
            dataset_transf_train = model.transform(dataset.train_dataset[0])
            dataset_transf_test = model.transform(dataset.test_dataset[0])

            print(classification_report(dataset.test_dataset[0].labels, dataset_transf_test.labels))

        else:
            raise ValueError("Unsupported model type")

        # benchmark = FairnessBenchmark(dataset, args.label, args.sensitive, model)
        # benchmark = FairnessBenchmark()
        # results = benchmark.run()
    else:
        logging.info("Domain Generalization task is not yet implemented.")
        return
    
    logging.info("Final Evaluation Results:")
    # for metric, value in results.items():
    #     logging.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
