# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd as autograd
import numpy as np
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

class IRM(torch.nn.Module):
    """Introduce

    References:
        .. VMartin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. 2019.
            Invariant risk minimization. arXiv preprint arXiv:1907.02893 (2019).
    """

    def __init__(self, batch_size, epoch, n_steps, irm_lambda=1e2, irm_penalty_anneal_iters=1000,
                 num_classes=2,lr=5e-5,weight_decay=0,):
        super().__init__()
        self.register_buffer('update_count', torch.tensor([0]))

        self.network = models.resnet50(pretrained=True)
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(in_features, num_classes)
        self.lr = lr
        self.irm_lambda = irm_lambda
        self.irm_penalty_anneal_iters = irm_penalty_anneal_iters
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.batch_size = batch_size
        self.epoch = epoch
        self.n_steps_per_epoch = n_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device) # (batch_size, channels, height, width) 

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, data, labels):
        penalty_weight = (self.irm_lambda if self.update_count
                          >= self.irm_penalty_anneal_iters else
                          1.0)
        nll = 0.
        penalty = 0.

        all_logits = self.network(data)
        all_logits_idx = 0
        batch_len = len(labels)
        for i in range(batch_len):
            x = data[i]
            y = labels[i].unsqueeze(0)
            logits = all_logits[all_logits_idx:all_logits_idx + 1]
            all_logits_idx += 1
            nll += F.cross_entropy(logits, y)
            
            penalty += self._irm_penalty(logits, y)
        nll /= batch_len
        penalty /= batch_len
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.irm_penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}
    
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

            print(f"Epoch [{epoch+1}/{self.epoch}], Loss: {total_loss:.4f}")

                    
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

    