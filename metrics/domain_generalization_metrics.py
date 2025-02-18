from sklearn.metrics import accuracy_score, f1_score


class DomainGeneralizationMetric:
    """
    Class for computing domain generalization metrics 
    """
    def __init__(self, labels, predicted_labels):
        self.labels = labels
        self.predicted_labels = predicted_labels

    def accuracy(self):

        return accuracy_score(self.labels, self.predicted_labels)

    def f1(self, average='binary'):

        return f1_score(self.labels, self.predicted_labels, average=average)
