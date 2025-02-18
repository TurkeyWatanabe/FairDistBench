# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader


from .. import networks

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ERM(torch.nn.Module):
    """Introduce

    References:
        .. Vapnik V. The nature of statistical learning theory[M]. Springer science & business media, 2013.
    """

    def __init__(self, batch_size, epoch, num_classes=2,lr=5e-5,weight_decay=0):
        super().__init__()
        
        # self.featurizer = networks.Featurizer(input_shape)
        # self.classifier = networks.Classifier(
        #     self.featurizer.n_outputs,
        #     num_classes,
        #     nonlinear_classifier)
        # self.network = nn.Sequential(self.featurizer, self.classifier)

        self.network = models.resnet50(pretrained=True)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.batch_size = batch_size
        self.epoch = epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device) # (batch_size, channels, height, width) 


    def update(self, data, labels):
        loss = F.cross_entropy(self.network(data), labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    
    def fit(self, dataset):
        """
        Model training.

        input: training dataset
        """
        custom_dataset = CustomDataset(dataset.data, dataset.labels)
        train_loader = DataLoader(custom_dataset, batch_size=self.batch_size, shuffle=True)

        for e in range(self.epoch):
            total_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device).permute(0, 3, 1, 2), batch_labels.to(self.device)
                res = self.update(batch_data, batch_labels)
                
                total_loss += res['loss']

            print(f"Epoch [{e+1}/{self.epoch}], Loss: {total_loss:.4f}")

                    
    def predict(self, dataset):
        """
        Model eval.

        input: test dataset
        output: predicted labels
        """

        self.network.eval()
        all_preds = []
        all_labels = []

        custom_dataset = CustomDataset(dataset.data, dataset.labels)
        test_loader = DataLoader(custom_dataset, batch_size=self.batch_size, shuffle=True)

        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(self.device).permute(0, 3, 1, 2), batch_labels.to(self.device)

                outputs = self.network(batch_data)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
            
        return all_preds, all_labels

    