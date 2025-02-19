import numpy as np
from sklearn.svm import OneClassSVM
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

class OCSVM:
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale'):
        """
        Initialize One-Class SVM for OOD detection.
        :param nu: An upper bound on the fraction of margin errors and a lower bound of support vectors.
        :param kernel: Specifies the kernel type to be used in the algorithm.
        :param gamma: Kernel coefficient.
        """
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True).to(self.device)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()
    
    def fit(self, dataset):
        """
        Train the One-Class SVM model using ID data only.
        :param dataset: A dataset object with dataset.data (images) and dataset.ood_labels (0 for ID, 1 for OOD)
        """

        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        dataloader = DataLoader(data, batch_size=bs, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        features = torch.cat(features_list, dim=0).numpy() 
        
        self.model.fit(features)
    
    def predict(self, dataset):
        """
        Predict OOD labels on the test dataset.
        :param dataset: A dataset object with dataset.data and dataset.ood_labels
        :return: Predictions and ground truth OOD labels
        """
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        dataloader = DataLoader(data, batch_size=bs, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        test_features = torch.cat(features_list, dim=0).numpy() 
        
        # Predict using the One-Class SVM (returns 1 for inliers and -1 for outliers)
        preds = self.model.decision_function(test_features)
        
        # Evaluation
        y_true = dataset.ood_labels
        
        # print(f'Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
        
        return preds, y_true