import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

class DDU:
    def __init__(self, task, epochs=5, batch_size=64, num_class=2, threshold_percent=90):
        # Initialize parameters and model
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percent = threshold_percent

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_class)
        self.resnet50 = self.resnet50.to(self.device)

        self.resnet_trained = False
        self.feature_extractor = None
        self.ddu = None
        self.gaussians_model = None
        self.jitter_eps = None

    def train_featurizer(self, dataset, epochs, batch_size, lr=1e-4):
        # Train the classifier using dataset.data and dataset.labels
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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        self.resnet_trained = True
        # Build feature extractor by removing the final fc layer
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.feature_extractor.eval()

    def fit(self, dataset):
        # Train the featurizer if not already trained
        if not self.resnet_trained:
            print("Training Featurizer and ID Classifier...")
            self.train_featurizer(dataset, self.epochs, self.batch_size)
        
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        labels = torch.tensor(dataset.labels).long()
        train_ds = TensorDataset(data, labels)
        dataloader = DataLoader(train_ds, batch_size=bs, shuffle=False)
        features_list = []
        labels_list = []
        self.resnet50.eval()
        with torch.no_grad():
            for inputs, lab in dataloader:
                inputs = inputs.to(self.device)
                features = self.feature_extractor(inputs)
                features = torch.flatten(features, 1)
                features_list.append(features.cpu())
                labels_list.append(lab)
        train_features = torch.cat(features_list, dim=0)
        train_labels = torch.cat(labels_list, dim=0)
        
        # Fit the GMM using an external gmm_fit function
        self.gaussians_model, self.jitter_eps = gmm_fit(embeddings=train_features, labels=train_labels, num_classes=train_labels.max().item()+1)
        
        # Compute training scores to set the threshold
        gmm_log_probs = self.gaussians_model.log_prob(train_features[:, None, :])
        scores = torch.logsumexp(gmm_log_probs, dim=1, keepdim=False).cpu().numpy()
        scores[np.isneginf(scores)] = 0
        self.ddu_threshold = np.percentile(scores, self.threshold_percent)
        print(f"DDU threshold set to: {self.ddu_threshold}")
    
    def predict(self, dataset):
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data_ds = TensorDataset(data)
        dataloader = DataLoader(data_ds, batch_size=bs, shuffle=False)
        features_list = []
        image_list = []
        self.resnet50.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                image_list.append(images)
                features = self.feature_extractor(images)
                features = torch.flatten(features, 1)
                features_list.append(features.cpu())
        test_features = torch.cat(features_list, dim=0)
        log_probs = self.gaussians_model.log_prob(test_features[:, None, :])
        test_scores = torch.logsumexp(log_probs, dim=1, keepdim=False).cpu().numpy()
        test_scores[np.isneginf(test_scores)] = 0
        
        # In DDU, higher score indicates in-distribution
        predicted_labels = (test_scores > self.ddu_threshold).astype(int)
        ood_true = np.array(dataset.ood_labels)
        
        if self.task == 'oodd-s':
            return test_scores, ood_true, predicted_labels, None, None
        elif self.task == 'oodd-a':
            id_indices = np.where(predicted_labels == 0)[0]
            id_true_labels = np.array(dataset.labels)[id_indices]
            
            all_images = torch.cat(image_list, dim=0)
            id_images = all_images[id_indices].to(self.device)
            
            self.resnet50.eval()
            with torch.no_grad():
                outputs = self.resnet50(id_images)
                _, id_predicted = torch.max(outputs, 1)
            id_predicted_labels = id_predicted.cpu().numpy()
            
            return test_scores, ood_true, predicted_labels, id_true_labels, id_predicted_labels
        else:
            raise ValueError(f"Unsupported task type")
        

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(
    net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device,
):
    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature

            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):

    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):

    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov_features = torch.stack(
            [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
        )
        # a = torch.isnan(classwise_mean_features).any()
        # b = torch.isnan(classwise_cov_features).any()
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
                elif "to satisfy the constraint PositiveDefinite(), but found invalid values" in str(e):
                    continue
                break

    return gmm, jitter_eps