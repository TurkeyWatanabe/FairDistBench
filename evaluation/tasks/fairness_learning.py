import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

logging.basicConfig(level=logging.INFO)

class FairnessBenchmark:
    def __init__(self, dataset):
        """
        :param dataset: pandas DataFrame, containing features, labels, and sensitive attributes
        :param label_column: str, target variable (label) column name
        :param sensitive_column: str, sensitive attribute column name
        :param model: sklearn-compatible classification model
        """
        self.dataset = dataset
        # self.label_column = label_column
        # self.sensitive_column = sensitive_column
        # self.model = model
        # self._prepare_data()
    
    
    def train_model(self):
        """Trains the model"""
        self.model.fit(self.X_train, self.y_train)
    
    # def evaluate(self):
    #     """Computes metrics"""
    #     y_pred = self.model.predict(self.X_test)
    #     accuracy = accuracy_score(self.y_test, y_pred)
        
    #     delta_dp = demographic_parity_difference(self.y_test, y_pred, sensitive_features=self.sensitive_test)
    #     delta_eo = equalized_odds_difference(self.y_test, y_pred, sensitive_features=self.sensitive_test)
        
    #     return {
    #         "Accuracy": accuracy,
    #         "Delta_DP": delta_dp,
    #         "Delta_EO": delta_eo
    #     }
    
    def run(self):
        """Executes the full evaluation process"""
        # self.train_model()
        # metrics = self.evaluate()
        # logging.info("Fairness Evaluation Metrics:")
        # for metric, value in metrics.items():
        #     logging.info(f"{metric}: {value:.4f}")
        # return metrics
        print('hello')
