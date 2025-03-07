# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader, RandomSampler


from .. import networks

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Mixup(torch.nn.Module):
    """Introduce

    References:
        .. Shen Yan, Huan Song, Nanxiang Li, Lincan Zou, and Liu Ren. 2020. Improve unsu-
        pervised domain adaptation with mixup training. arXiv preprint arXiv:2001.00677 (2020).
    """

    def __init__(self, batch_size, epoch, n_steps, mixup_alpha=0.2, num_classes=2,lr=5e-5,weight_decay=0):
        super().__init__()

        self.network = models.resnet50(pretrained=True)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)
        self.mixup_alpha = mixup_alpha

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.batch_size = batch_size
        self.epoch = epoch
        self.n_steps_per_epoch = n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device) # (batch_size, channels, height, width) 

    @staticmethod
    def random_pairs_of_minibatches(data, labels):
        batch_len = len(labels)
        perm = torch.randperm(batch_len).tolist()
        pairs = []

        for i in range(batch_len):
            j = i + 1 if i < (batch_len - 1) else 0

            xi, yi = data[perm[i]].unsqueeze(0), labels[perm[i]].unsqueeze(0)
            xj, yj = data[perm[j]].unsqueeze(0), labels[perm[j]].unsqueeze(0)

            min_n = min(len(xi), len(xj))

            pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

        return pairs

    def update(self, data, labels):
        objective = 0

        for (xi,yi), (xj,yj) in self.random_pairs_of_minibatches(data, labels):
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            x = lam * xi + (1 - lam) * xj
            predictions = self.network(x)

            objective += lam * F.cross_entropy(predictions, yi)
            objective += (1 - lam) * F.cross_entropy(predictions, yj)

        objective /= len(labels)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item()}
    
    def fit(self, dataset):
        """
        Model training.

        input: training dataset
        """
        train_data = CustomDataset(dataset.data, dataset.labels)
        random_sampler = RandomSampler(train_data, replacement=True)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=random_sampler)

        for epoch in range(self.epoch):
            total_loss = 0.0
            for step in range(self.n_steps_per_epoch):
                batch_data, batch_labels = next(iter(train_loader))
                batch_data, batch_labels = batch_data.to(self.device).permute(0, 3, 1, 2), batch_labels.to(self.device)

                res = self.update(batch_data, batch_labels)
                
                total_loss += res['loss']

            logging.info(f"Epoch [{epoch+1}/{self.epoch}], Loss: {total_loss:.4f}")

                    
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

    