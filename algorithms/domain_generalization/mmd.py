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

class MMD(torch.nn.Module):
    """Introduce

    References:
        .. Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. 2018. Domain gener-
        alization with adversarial feature learning. In Proceedings of the IEEE conference
        on computer vision and pattern recognition. 5400–5409.
    """

    def __init__(self, batch_size, epoch, n_steps, mmd_gamma=1., num_classes=2,lr=5e-5,weight_decay=0):
        super().__init__()

        gaussian = True
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

        resnet = models.resnet50(pretrained=True)
        self.featurizer = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features
        self.classifier = nn.Linear(in_features, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.mmd_gamma = mmd_gamma

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

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
        

    def update(self, data, labels):
        objective = 0
        penalty = 0
        batch_len= len(labels)
        nmb = batch_len

        features = [self.featurizer(xi.unsqueeze(0)).squeeze(-1).squeeze(-1) for xi in data]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi.unsqueeze(0) for yi in labels]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.mmd_gamma * penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}
    
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

    