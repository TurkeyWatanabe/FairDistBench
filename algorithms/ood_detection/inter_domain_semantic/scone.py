import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def to_np(x): return x.data.cpu().numpy()

class SCONE:
    def __init__(self, epochs = 5, batch_size = 64,num_classes=2, threshold_percent=90,
                 false_alarm_cutoff=0.05, in_constraint_weight=1, ce_constraint_weight=1,
                 ce_tol=2, penalty_mult=2.5, lr_lam=1):
        """
        SCONE
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percent = threshold_percent
        self.false_alarm_cutoff = false_alarm_cutoff
        self.in_constraint_weight = in_constraint_weight
        self.ce_constraint_weight = ce_constraint_weight
        self.ce_tol = ce_tol
        self.penalty_mult = penalty_mult
        self.lr_lam = lr_lam
        self.num_classes = num_classes

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)
        self.resnet50 = self.resnet50.to(self.device)

        self.logistic_regression = nn.Linear(1, 1)
        self.logistic_regression = self.logistic_regression.to(self.device)

        self.resnet_trained = False   # Flag to check if ResNet classifier is trained
        self.feature_extractor = None # Model for feature extraction after training
        self.energy_threshold = None

    
    def compute_energy(self, logits, T=1.0):
        # Compute energy using the logits
        return -T * torch.logsumexp(logits / T, dim=1)
    
    def evaluate_classification_loss_training(self, data_loader):
        '''
        evaluate classification loss on training dataset
        '''

        self.resnet50.eval()
        losses = []
        for inputs, target in data_loader:
            data, target = inputs.to(self.device), target.to(self.device)
            # forward
            x = self.resnet50(data)

            # in-distribution classification accuracy
            x_classification = x[:, :self.num_classes]
            loss_ce = F.cross_entropy(x_classification, target, reduction='none')

            losses.extend(list(to_np(loss_ce)))

        avg_loss = np.mean(np.array(losses))
        print("average loss fr classification {}".format(avg_loss))

        return avg_loss
    
    def evaluate_energy_logistic_loss(self, train_loader):
        '''
        evaluate energy logistic loss on training dataset
        '''

        self.resnet50.eval()
        self.logistic_regression.eval()
        sigmoid_energy_losses = []
        logistic_energy_losses = []
        ce_losses = []
        for inputs, target in train_loader:
            data, target = inputs.to(self.device), target.to(self.device)

            # forward
            x = self.resnet50(data)

            # compute energies
            Ec_in = torch.logsumexp(x, dim=1)

            # compute labels
            binary_labels_1 = torch.ones(len(data)).to(self.device)

            # compute in distribution logistic losses
            logistic_loss_energy_in = F.binary_cross_entropy_with_logits(self.logistic_regression(
                Ec_in.unsqueeze(1)).squeeze(), binary_labels_1, reduction='none')

            logistic_energy_losses.extend(list(to_np(logistic_loss_energy_in)))

            # compute in distribution sigmoid losses
            sigmoid_loss_energy_in = torch.sigmoid(self.logistic_regression(
                Ec_in.unsqueeze(1)).squeeze())

            sigmoid_energy_losses.extend(list(to_np(sigmoid_loss_energy_in)))

            # in-distribution classification losses
            x_classification = x[:, :self.num_classes]
            loss_ce = F.cross_entropy(x_classification, target, reduction='none')

            ce_losses.extend(list(to_np(loss_ce)))

        avg_sigmoid_energy_losses = np.mean(np.array(sigmoid_energy_losses))
        print("average sigmoid in distribution energy loss {}".format(avg_sigmoid_energy_losses))

        avg_logistic_energy_losses = np.mean(np.array(logistic_energy_losses))
        print("average in distribution energy loss {}".format(avg_logistic_energy_losses))

        avg_ce_loss = np.mean(np.array(ce_losses))
        print("average loss fr classification {}".format(avg_ce_loss))

        return avg_sigmoid_energy_losses, avg_logistic_energy_losses, avg_ce_loss


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
        lam = torch.tensor(0).float()
        lam = lam.to(self.device)

        lam2 = torch.tensor(0).float()
        lam2 = lam.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.resnet50.parameters(), lr=lr)
        
        self.resnet50.train()
        self.logistic_regression.train()
        full_train_loss = self.evaluate_classification_loss_training(dataloader)
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, target in dataloader:
                inputs, target = inputs.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.resnet50(inputs)
                loss_ce = criterion(outputs, target)

                #apply the sigmoid loss
                loss_energy_in =  torch.mean(torch.sigmoid(self.logistic_regression(
                    (torch.logsumexp(outputs, dim=1)).unsqueeze(1)).squeeze()))

                #alm function for the in distribution constraint
                in_constraint_term = loss_energy_in - self.false_alarm_cutoff
                if self.in_constraint_weight * in_constraint_term + lam >= 0:
                    in_loss = in_constraint_term * lam + self.in_constraint_weight / 2 * torch.pow(in_constraint_term, 2)
                else:
                    in_loss = - torch.pow(lam, 2) * 0.5 / self.in_constraint_weight

                #alm function for the cross entropy constraint
                loss_ce_constraint = loss_ce - self.ce_tol * full_train_loss
                if self.ce_constraint_weight * loss_ce_constraint + lam2 >= 0:
                    loss_ce = loss_ce_constraint * lam2 + self.ce_constraint_weight / 2 * torch.pow(loss_ce_constraint, 2)
                else:
                    loss_ce = - torch.pow(lam2, 2) * 0.5 / self.ce_constraint_weight
                
                loss_ce = loss_ce.clone().detach().requires_grad_()

                loss = loss_ce + in_loss

                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            print("making updates for energy alm methods...")
            avg_sigmoid_energy_losses, _, avg_ce_loss = self.evaluate_energy_logistic_loss(dataloader)

            in_term_constraint = avg_sigmoid_energy_losses -  self.false_alarm_cutoff
            print("in_distribution constraint value {}".format(in_term_constraint))

            # update lambda
            print("updating lam...")
            if in_term_constraint * self.in_constraint_weight + lam >= 0:
                lam += self.lr_lam * in_term_constraint
            else:
                lam += -self.lr_lam * lam / self.in_constraint_weight

            print("making updates for energy alm methods...")
            avg_sigmoid_energy_losses, _, avg_ce_loss = self.evaluate_energy_logistic_loss(dataloader)

            in_term_constraint = avg_sigmoid_energy_losses -  self.false_alarm_cutoff
            print("in_distribution constraint value {}".format(in_term_constraint))

            print("updating lam2...")

            ce_constraint = avg_ce_loss - self.ce_tol * full_train_loss
            print("cross entropy constraint {}".format(ce_constraint))

            # update lambda2
            if ce_constraint * self.ce_constraint_weight + lam2 >= 0:
                lam2 += self.lr_lam * ce_constraint
            else:
                lam2 += -self.lr_lam * lam2 / self.ce_constraint_weight

            
            self.in_constraint_weight *= self.penalty_mult
            self.ce_constraint_weight *= self.penalty_mult

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
        energy_list = []
        self.resnet50.eval()
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                logits = self.resnet50(inputs)
                energy = self.compute_energy(logits)
                energy_list.append(energy.cpu())
        energies = torch.cat(energy_list, dim=0).numpy()
        
        # Set energy threshold based on the self.threshold_percent
        self.energy_threshold = np.percentile(energies, self.threshold_percent)
        logging.info(f"Energy threshold set to: {self.energy_threshold}")
    
    def predict(self, dataset):
        """
        Predict OOD labels on the test dataset.
        :param dataset: A dataset object with dataset.data and dataset.ood_labels
        :return: Predictions and ground truth OOD labels
        """
        bs = 256
        data = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()
        data_ds = TensorDataset(data)
        dataloader = DataLoader(data_ds, batch_size=bs, shuffle=False)
        energy_list = []
        logit_list = []
        self.resnet50.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                logits = self.resnet50(images)
                logit_list.append(logits.cpu())
                energy = self.compute_energy(logits)
                energy_list.append(energy.cpu())
        test_energy = torch.cat(energy_list, dim=0).numpy()
        
        energy_predicted_labels = (test_energy > self.energy_threshold).astype(int)
        ood_true = np.array(dataset.ood_labels)
        

        id_indices = np.where(energy_predicted_labels == 0)[0]
        id_true_labels = np.array(dataset.labels)[id_indices]
        
        logit_list = torch.cat(logit_list, dim=0).cpu().numpy()
        id_logits = logit_list[id_indices]
        
        id_predicted = np.argmax(id_logits, axis=1)
        id_predicted_labels = id_predicted
        
        return test_energy, ood_true, energy_predicted_labels, id_true_labels, id_predicted_labels


        