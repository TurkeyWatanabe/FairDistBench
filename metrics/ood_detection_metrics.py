from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


class OODDetectionMetrics:
    """
    Class for computing ood detection metrics 

    ood_labels: 0-ID, 1-OOD
    ood_label_preds: probabilities
    id_labels: 0-class1, 1-class2
    id_class_label_preds: probabilities

    relation:
    if ood_label == 1 --> OOD
    elif ood_label == 0 --> ID
        if id_class_label == 0 --> ID-Class 1
        elif id_class_label == 1 --> ID-Class 2
    """
    def __init__(self, ood_labels, ood_preds, ood_predicted_labels, id_class_labels=None, id_class_predicted_labels=None):
        self.ood_labels = ood_labels
        self.ood_preds = ood_preds
        self.ood_predicted_labels = ood_predicted_labels
        self.id_class_labels = id_class_labels
        self.id_class_predicted_labels = id_class_predicted_labels


    def ood_id_accuracy(self):
        # only for inter-domian sensory ood detection

        return accuracy_score(self.ood_labels, self.ood_predicted_labels)
    
    def id_accuracy(self):

        return accuracy_score(self.id_class_labels, self.id_class_predicted_labels)
    
    
    def auroc(self):

        return roc_auc_score(self.ood_labels, self.ood_preds)
    
    def aupr(self):

        return average_precision_score(self.ood_labels, self.ood_preds)
