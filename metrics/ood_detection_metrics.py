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
    def __init__(self, model, ood_labels, ood_label_preds, id_class_labels=None, id_class_label_preds=None):
        self.ood_labels = ood_labels
        self.ood_label_preds = ood_label_preds
        self.id_class_labels = id_class_labels
        self.id_class_label_preds = id_class_label_preds

        self.model = model

    def ood_id_accuracy(self):
        # only for inter-domian sensory ood detection
        if self.model == 'oc-svm':
            preds = (self.ood_label_preds < 0).astype(int)

        return accuracy_score(self.ood_labels, preds)
    
    def id_accuracy(self):

        return accuracy_score(self.id_class_labels, (self.id_class_label_preds>=0.5).astype(int))
    
    
    def auroc(self):

        return roc_auc_score(self.ood_labels, self.ood_label_preds)
    
    def aupr(self):

        return average_precision_score(self.ood_labels, self.ood_label_preds)
