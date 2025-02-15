# Based on code from https://github.com/Trusted-AI/AIF360/tree/main

import numpy as np
import scipy.optimize as optim
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset


class LFR:
    """Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.
    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.
    Based on code from https://github.com/Trusted-AI/AIF360/tree/main
    """

    def __init__(self,
                 k=5,
                 Ax=0.01,
                 Ay=1.0,
                 Az=50.0,
                 print_interval=250,
                 verbose=0,
                 seed=None):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged group.
            privileged_groups (tuple): Representation for privileged group.
            k (int, optional): Number of prototypes.
            Ax (float, optional): Input recontruction quality term weight.
            Az (float, optional): Fairness constraint term weight.
            Ay (float, optional): Output prediction error.
            print_interval (int, optional): Print optimization objective value
                every print_interval iterations.
            verbose (int, optional): If zero, then no output.
            seed (int, optional): Seed to make `predict` repeatable.
        """
        self.seed = seed

        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az

        self.print_interval = print_interval
        self.verbose = verbose

        self.w = None
        self.prototypes = None
        self.learned_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = models.resnet50(pretrained=True).to(self.device)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
        self.resnet50.eval()

    def fit(self, dataset, maxiter=5000, maxfun=5000):
        """Compute the transformation parameters that leads to fair representations.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
            maxiter (int): Maximum number of iterations.
            maxfun (int): Maxinum number of function evaluations.
        Returns:
            LFR: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        

        self.features_dim = 2048 # Decided by resnet50

        unprivileged_sample_ids = np.where(dataset.sensitive_attribute == 0)[0]
        privileged_sample_ids = np.where(dataset.sensitive_attribute == 1)[0]
        data_unprivileged = dataset.data[unprivileged_sample_ids]
        data_privileged = dataset.data[privileged_sample_ids]

        batch_size = 256
        data_unprivileged = torch.tensor(data_unprivileged).permute(0, 3, 1, 2).float()
        data_unprivileged = TensorDataset(data_unprivileged)
        dataloader_unprivileged = DataLoader(data_unprivileged, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader_unprivileged:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        features_unprivileged = torch.cat(features_list, dim=0).numpy()

        data_privileged = torch.tensor(data_privileged).permute(0, 3, 1, 2).float()
        data_privileged = TensorDataset(data_privileged)
        dataloader_privileged = DataLoader(data_privileged, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader_privileged:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        features_privileged = torch.cat(features_list, dim=0).numpy()

        labels_unprivileged = dataset.labels[unprivileged_sample_ids]
        labels_privileged = dataset.labels[privileged_sample_ids]

        # Initialize the LFR optim objective parameters
        parameters_initialization = np.random.uniform(size=self.k + self.features_dim * self.k)
        bnd = [(0, 1)]*self.k + [(None, None)]*self.features_dim*self.k
        LFR_optim_objective.steps = 0

        print('Start train LFR...')
        self.learned_model = optim.fmin_l_bfgs_b(LFR_optim_objective, x0=parameters_initialization, epsilon=1e-5,
                                                      args=(features_unprivileged, features_privileged,
                                        labels_unprivileged, labels_privileged, self.k, self.Ax,
                                        self.Ay, self.Az, self.print_interval, self.verbose),
                                                      bounds=bnd, approx_grad=True, maxfun=maxfun,
                                                      maxiter=maxiter, disp=self.verbose)[0]
        self.w = self.learned_model[:self.k]
        self.prototypes = self.learned_model[self.k:].reshape((self.k, self.features_dim))

        return self

    def transform(self, dataset, threshold=0.5):
        """Transform the dataset using learned model parameters.
        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs to be transformed.
            threshold(float, optional): threshold parameter used for binary label prediction.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        unprivileged_sample_ids = np.where(dataset.sensitive_attribute == 0)[0]
        privileged_sample_ids = np.where(dataset.sensitive_attribute == 1)[0]
        data_unprivileged = dataset.data[unprivileged_sample_ids]
        data_privileged = dataset.data[privileged_sample_ids]

        batch_size = 256
        data_unprivileged = torch.tensor(data_unprivileged).permute(0, 3, 1, 2).float()
        data_unprivileged = TensorDataset(data_unprivileged)
        dataloader_unprivileged = DataLoader(data_unprivileged, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader_unprivileged:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        features_unprivileged = torch.cat(features_list, dim=0).numpy() 

        data_privileged = torch.tensor(data_privileged).permute(0, 3, 1, 2).float()
        data_privileged = TensorDataset(data_privileged)
        dataloader_privileged = DataLoader(data_privileged, batch_size=batch_size, shuffle=False)
        features_list = []
        with torch.no_grad():
            for batch in dataloader_privileged:
                batch = batch[0].to(self.device) 
                features = self.resnet50(batch) 
                features_list.append(features.squeeze().cpu()) 
        features_privileged = torch.cat(features_list, dim=0).numpy()

        _, features_hat_unprivileged, labels_hat_unprivileged = get_xhat_y_hat(self.prototypes, self.w, features_unprivileged)

        _, features_hat_privileged, labels_hat_privileged = get_xhat_y_hat(self.prototypes, self.w, features_privileged)

        print(features_hat_unprivileged.shape)
        print(features_hat_privileged.shape)

        transformed_features = np.zeros(shape=(len(dataset.labels),2048))
        transformed_labels = np.zeros(shape=np.shape(dataset.labels))
        transformed_features[unprivileged_sample_ids] = features_hat_unprivileged
        transformed_features[privileged_sample_ids] = features_hat_privileged
        transformed_labels[unprivileged_sample_ids] = np.reshape(labels_hat_unprivileged, [-1, 1])
        transformed_labels[privileged_sample_ids] = np.reshape(labels_hat_privileged,[-1, 1])
        transformed_bin_labels = (np.array(transformed_labels) > threshold).astype(np.float64)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.features = transformed_features
        dataset_new.labels = transformed_bin_labels
        # dataset_new.scores = np.array(transformed_labels)

        return dataset_new

    def fit_transform(self, dataset, maxiter=5000, maxfun=5000, threshold=0.5):
        """Fit and transform methods sequentially.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs to be transformed.
            maxiter (int): Maximum number of iterations.
            maxfun (int): Maxinum number of function evaluations.
            threshold(float, optional): threshold parameter used for binary label prediction.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.
        """

        return self.fit(dataset, maxiter=maxiter, maxfun=maxfun).transform(dataset, threshold=threshold)



def LFR_optim_objective(parameters, x_unprivileged, x_privileged, y_unprivileged,
                        y_privileged, k=10, A_x=0.01, A_y=0.1, A_z=0.5, print_interval=250, verbose=1):

    features_dim = 2048

    w = parameters[:k]
    prototypes = parameters[k:].reshape((k, features_dim))

    M_unprivileged, x_hat_unprivileged, y_hat_unprivileged = get_xhat_y_hat(prototypes, w, x_unprivileged)

    M_privileged, x_hat_privileged, y_hat_privileged = get_xhat_y_hat(prototypes, w, x_privileged)

    y_hat = np.concatenate([y_hat_unprivileged, y_hat_privileged], axis=0)
    y = np.concatenate([y_unprivileged.reshape((-1, 1)), y_privileged.reshape((-1, 1))], axis=0)

    L_x = np.mean((x_hat_unprivileged - x_unprivileged) ** 2) + np.mean((x_hat_privileged - x_privileged) ** 2)
    L_z = np.mean(abs(np.mean(M_unprivileged, axis=0) - np.mean(M_privileged, axis=0)))
    L_y = - np.mean(y * np.log(y_hat) + (1. - y) * np.log(1. - y_hat))

    total_loss = A_x * L_x + A_y * L_y + A_z * L_z

    # if verbose and LFR_optim_objective.steps % print_interval == 0:
    print("step: {}, loss: {}, L_x: {},  L_y: {},  L_z: {}".format(
            LFR_optim_objective.steps, total_loss, L_x,  L_y,  L_z))
    LFR_optim_objective.steps += 1

    return total_loss


def get_xhat_y_hat(prototypes, w, x):
    M = softmax(-cdist(x, prototypes), axis=1)
    x_hat = np.matmul(M, prototypes)
    y_hat = np.clip(
        np.matmul(M, w.reshape((-1, 1))),
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps
    )
    return M, x_hat, y_hat