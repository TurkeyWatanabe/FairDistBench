import numpy as np



class BinaryLabelFairnessMetric:
    """
    Class for computing fairness metrics 
    """
    def __init__(self, label, predicted_label, sensitive_attribute):
        self.label = label
        self.predicted_label = predicted_label
        self.sensitive_attribute = sensitive_attribute

    def accuracy(self):
        """
        Compute the accuracy of the predictions.
        
        Parameters:
        - label: Ground truth labels (numpy array)
        - predicted_label: Predicted labels (numpy array)
        
        Returns:
        - accuracy: The proportion of correctly predicted labels
        """

        return np.mean(self.label == self.predicted_label)

    def difference_DP(self):
        """
        Compute the Difference in Demographic Parity (DP).
        
        Demographic Parity measures whether different sensitive groups 
        receive positive predictions at the same rate.
        
        Parameters:
        - predicted_label: Predicted labels (numpy array)
        - sensitive_attribute: Sensitive attribute values (numpy array), assumed to have two groups (0 and 1)
        
        Returns:
        - difference_DP: Absolute difference in positive prediction rates between the two groups
        """
        group_0 = self.predicted_label[self.sensitive_attribute == 0]
        group_1 = self.predicted_label[self.sensitive_attribute == 1]
        
        p_positive_0 = np.mean(group_0 == 1)  # Probability of positive prediction in group 0
        p_positive_1 = np.mean(group_1 == 1)  # Probability of positive prediction in group 1
        
        return abs(p_positive_0 - p_positive_1)

    def difference_EO(self):
        """
        Compute the Difference in Equal Opportunity (EO).
        
        Equal Opportunity measures whether different sensitive groups 
        have equal True Positive Rates (TPR), meaning they receive 
        positive predictions at the same rate when the true label is 1.
        
        Parameters:
        - label: Ground truth labels (numpy array)
        - predicted_label: Predicted labels (numpy array)
        - sensitive_attribute: Sensitive attribute values (numpy array), assumed to have two groups (0 and 1)
        
        Returns:
        - difference_EO: Absolute difference in TPR between the two groups
        """
        # Select only samples where the ground truth label is 1
        positive_indices = (self.label == 1)
        
        group_0 = self.predicted_label[(self.sensitive_attribute == 0) & positive_indices]
        group_1 = self.predicted_label[(self.sensitive_attribute == 1) & positive_indices]
        
        tpr_0 = np.mean(group_0 == 1)  # True Positive Rate for group 0
        tpr_1 = np.mean(group_1 == 1)  # True Positive Rate for group 1
        
        return abs(tpr_0 - tpr_1)
