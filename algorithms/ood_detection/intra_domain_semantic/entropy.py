import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Entropy:
    def __init__(self, task, epochs=5, batch_size=64, num_class=2, threshold_percent=90, beta=0.2):
        # Initialize parameters and model
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percent = threshold_percent
        self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_class)
        self.resnet50 = self.resnet50.to(self.device)

        self.resnet_trained = False
        self.feature_extractor = None
        self.entropy_threshold = None

    def compute_entropy(self, logits):
        # Compute entropy from logits
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        return entropy

    def train_featurizer(self, dataset, epochs, batch_size, lr=1e-4):
        # Train classifier using dataset.data and dataset.labels
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        labels = torch.tensor(dataset.labels).long()
        division = data.shape[0]//2
        id_data = data[0:division]
        id_labels = labels[0:division]
        ood_data = data[division:]
        ood_labels = labels[division:]
        train_ds_id = TensorDataset(id_data, id_labels)
        id_dataloader = DataLoader(train_ds_id, batch_size=batch_size, shuffle=True)
        train_ds_ood = TensorDataset(ood_data, ood_labels)
        ood_dataloader = DataLoader(train_ds_ood, batch_size=batch_size, shuffle=True)
        ood_train_iter = iter(ood_dataloader)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.resnet50.parameters(), lr=lr)
        
        self.resnet50.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target in id_dataloader:
                data_ood, _ = next(ood_train_iter)
                ood_inputs = data_ood.to(self.device)
                id_inputs, id_target = inputs.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()

                output_id = self.resnet50(id_inputs)
                E_id = -torch.mean(torch.sum(F.log_softmax(output_id, dim=1) * F.softmax(output_id, dim=1), dim=1))

                output_ood = self.resnet50(ood_inputs)
                E_ood = -torch.mean(torch.sum(F.log_softmax(output_ood, dim=1) * F.softmax(output_ood, dim=1), dim=1))
                
                # Entropy based Margin-Loss
                loss = criterion(output_id, id_target) + self.beta * torch.clamp(0.4 + E_id - E_ood, min=0)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(train_ds_id)
            logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        self.resnet_trained = True
        # After training, construct the feature extractor by removing the final fully connected layer
        self.feature_extractor = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.feature_extractor.eval()

    def fit(self, dataset):
        # Train the featurizer if not already trained
        if not self.resnet_trained:
            logging.info("Training Featurizer and ID Classifier...")
            self.train_featurizer(dataset, self.epochs, self.batch_size)
        
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data = TensorDataset(data)
        dataloader = DataLoader(data, batch_size=bs, shuffle=False)
        entropy_list = []
        self.resnet50.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                logits = self.resnet50(inputs)
                entropy = self.compute_entropy(logits)
                entropy_list.append(entropy.cpu())
        entropies = torch.cat(entropy_list, dim=0).numpy()
        
        # Set threshold based on the threshold_percent percentile
        self.entropy_threshold = np.percentile(entropies, self.threshold_percent)
        logging.info(f"Entropy threshold set to: {self.entropy_threshold}")
    
    def predict(self, dataset):
        # Predict OOD labels on the test dataset
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data_ds = TensorDataset(data)
        dataloader = DataLoader(data_ds, batch_size=bs, shuffle=False)
        entropy_list = []
        logit_list = []
        self.resnet50.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                logits = self.resnet50(images)
                logit_list.append(logits.cpu())
                entropy = self.compute_entropy(logits)
                entropy_list.append(entropy.cpu())
        test_entropy = torch.cat(entropy_list, dim=0).numpy()
        
        entropy_predicted_labels = (test_entropy > self.entropy_threshold).astype(int)
        ood_true = np.array(dataset.ood_labels)
        
        if self.task == 'oodd-s':
            return test_entropy, ood_true, entropy_predicted_labels, None, None
        elif self.task == 'oodd-a':
            id_indices = np.where(entropy_predicted_labels == 0)[0]
            id_true_labels = np.array(dataset.labels)[id_indices]
            
            logit_list = torch.cat(logit_list, dim=0).numpy()
            id_logits = logit_list[id_indices]
            
            id_predicted = np.argmax(id_logits, axis=1)
            id_predicted_labels = id_predicted
            
            return test_entropy, ood_true, entropy_predicted_labels, id_true_labels, id_predicted_labels
        else:
            raise ValueError(f"Unsupported task type")