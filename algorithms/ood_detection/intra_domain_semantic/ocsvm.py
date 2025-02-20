import numpy as np
import logging
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

class OCSVM:
    def __init__(self, task, epochs = 5, batch_size = 64,num_class=2, nu=0.001, kernel='rbf', gamma=0.001):
        """
        Initialize One-Class SVM for OOD detection.
        :param nu: An upper bound on the fraction of margin errors and a lower bound of support vectors.
        :param kernel: Specifies the kernel type to be used in the algorithm.
        :param gamma: Kernel coefficient.
        :epochs: Number of epoch for featurizer training
        :batch_size: Size of batch for featurizer training
        """
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_class)
        self.resnet50 = self.resnet50.to(self.device)

        self.resnet_trained = False   # Flag to check if ResNet classifier is trained
        self.feature_extractor = None # Model for feature extraction after training
    
    def train_featurizer(self, dataset, epochs, batch_size, lr=1e-4):
        """
        Train the Featurizer classifier using dataset.data and dataset.labels.
        
        :param dataset: Dataset object containing dataset.data (images) and dataset.labels (class labels)
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        :param lr: Learning rate for the optimizer.
        """

        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        labels = torch.tensor(dataset.labels).long()
        train_ds = TensorDataset(data, labels)
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.resnet50.parameters(), lr=lr)
        
        self.resnet50.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target in dataloader:
                inputs, target = inputs.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.resnet50(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_ds)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        self.resnet_trained = True
        # After training, construct the feature extractor by removing the final fully connected layer
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.feature_extractor.eval()

    def fit(self, dataset):
        """
        Train the One-Class SVM model using ID data only.
        :param dataset: A dataset object with dataset.data (images) and dataset.ood_labels (0 for ID, 1 for OOD)
        """
        if not self.resnet_trained:
            logging.info("Training Featurizer and ID Classifier...")
            self.train_featurizer(dataset, self.epochs, self.batch_size)
        
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        dataloader = DataLoader(data, batch_size=bs, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device) 
                features = self.feature_extractor(batch) 
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
        logit_list = []
        self.resnet50.eval()
        self.feature_extractor.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device) 
                logits = self.resnet50(images)
                logit_list.append(logits)
                features = self.feature_extractor(images) 
                features_list.append(features.squeeze().cpu()) 
        test_features = torch.cat(features_list, dim=0).numpy() 
        
        svm_preds = self.model.decision_function(test_features)  # SVM decision function values
        
        ood_true = np.array(dataset.ood_labels)
        # Based on SVM decision function: < 0 is considered OOD (label 1), >= 0 is considered ID (label 0)
        svm_predicted_labels = (svm_preds < 0).astype(int)
        
        if self.task == 'oodd-s':
            return svm_preds, ood_true, svm_predicted_labels, None, None
        elif self.task == 'oodd-a':
            # For ID samples, perform classification prediction
            id_indices = np.where(svm_predicted_labels == 0)[0]
            id_true_labels = np.array(dataset.labels)[id_indices]
            
            logit_list = torch.cat(logit_list, dim=0).cpu().numpy()
            id_logits = logit_list[id_indices]
            
            id_predicted = np.argmax(id_logits, axis=1)
            id_predicted_labels = id_predicted
            
            return svm_preds, ood_true, svm_predicted_labels, id_true_labels, id_predicted_labels
        else:
            raise ValueError(f"Unsupported task type")


        