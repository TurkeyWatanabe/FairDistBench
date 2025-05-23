# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

import logging
from torch.utils.data import Dataset, DataLoader, RandomSampler, TensorDataset


class CustomDataset(Dataset):
    def __init__(self, data, labels, domains):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.domains = torch.tensor(domains, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.domains[idx]

def get_resnet():
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1]) 
    return resnet

def create_one_hot(y, classes, device):
    y_onehot = torch.LongTensor(y.size(0), classes).to(device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def get_sample_mixup_random(domains):
    indeces = torch.randperm(domains.size(0))
    return indeces.long()


def get_ratio_mixup_Dirichlet(domains, mixup_dir_list):
    RG = np.random.default_rng()
    return torch.from_numpy(
        RG.dirichlet(mixup_dir_list, size=domains.size(0))
    ).float()  # N * 3


def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions, dim=1), dim=1))
    return loss


def DistillKL(y_s, y_t, T):
    """KL divergence for distillation"""
    p_s = F.log_softmax(y_s / T, dim=1)
    p_t = y_t
    loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]
    return loss

class Ensemble_MMD_with_Distill(torch.nn.Module):
    """
    Take Face4FairShifts as an example: 4 domains in dataset --> 3 domains in training set
    For each domain, a featurizer is assigned
    
    """
    def __init__(self, num_domains =3 ,epochs=5, n_steps=1000,batch_size=64,num_classes=3, 
                 T=2.0, trade=3.0, trade2=1.0, trade3=1.0, trade4=3.0, mixup_dir=0.6, 
                 mixup_dir2=0.2, stop_gradient=1, meta_step_size=0.01, lr=1e-2):
        
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_domains = num_domains

        self.params = []
        self.featurizers = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for i in range(self.num_domains):
            f = get_resnet().to(self.device)
            c = nn.Linear(2048, num_classes).to(self.device)
            self.featurizers.append(f)
            self.classifiers.append(c)
            self.params.append({"params": f.parameters(), "lr": lr})
            self.params.append({"params": c.parameters(), "lr": lr})

        self.epoch = epochs
        self.n_steps_per_epoch = n_steps
        self.batch_size = batch_size
        self.bz = batch_size // self.num_domains
        self.num_classes = num_classes

        # hyper-parameters
        self.T = T
        self.trade = trade
        self.trade2 = trade2
        self.trade3 = trade3
        self.trade4 = trade4
        self.mixup_dir = mixup_dir
        self.mixup_dir2 = mixup_dir2
        self.stop_gradient = stop_gradient
        self.meta_step_size = meta_step_size
        self.kernel_type = "gaussian"

        self.opt = torch.optim.SGD(
            self.params,
            lr=lr,
            momentum=0.9,
            weight_decay=0.0,
            nesterov=True,
        )

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2
        ).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, data, labels, domains):
        nmb = self.num_domains  # num of domain
        cls_loss = 0
        penalty = 0
        mixup_loss = 0.0
        meta_train_loss = 0.0

        all_x = data
        all_y = labels

        X = []
        Y = []
        for i in range(self.num_domains):
            X.append(all_x[self.bz*i : self.bz*(i+1)])
            Y.append(all_y[self.bz*i : self.bz*(i+1)])

        all_domain = domains
        ###### Meta Train ##########
        for weight in self.parameters():
            weight.fast = None

        total_all_f_s = [[] for i in range(self.num_domains)]  # model_domains * 3batch_size
        all_one_hot_labels = []  # 3batch_size
        cnt = 0
        for data_domain, x_s_and_labels_s in enumerate(zip(X, Y)):
            x_s, labels_s = x_s_and_labels_s
            one_hot_labels = create_one_hot(labels_s, self.num_classes, self.device)
            all_one_hot_labels.append(one_hot_labels)

            # compute output
            y_s_distill = []
            for model_domain in range(nmb):
                f_s = self.featurizers[model_domain](x_s).squeeze(-1).squeeze(-1)
                y_s = self.classifiers[model_domain](f_s)
                total_all_f_s[model_domain].append(f_s)
                # y_s, f_s = model(x_s, domain=model_domain)
                if model_domain != data_domain:
                    y_s_distill.append(y_s)
                    cls_loss += F.cross_entropy(y_s, labels_s)
                else:
                    y_s_pred = y_s
                    cls_loss += 3 * F.cross_entropy(y_s, labels_s)

            meta_train_loss += cls_loss

            # Distill
            y_s_distill = torch.stack(y_s_distill)  # 2 * N * C
            y_s_distill = F.softmax(y_s_distill / self.T, dim=2)
            domains = [0] * y_s_distill.shape[1]
            domains = torch.LongTensor(domains)

            mixup_ratios = get_ratio_mixup_Dirichlet(domains, [1.0]*(nmb-1))
            mixup_ratios = mixup_ratios.to(self.device)  # N * 2
            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)  # 2 * N * 1
            y_s_distill = torch.sum(y_s_distill * mixup_ratios , dim=0)
            kd_loss = DistillKL(y_s_pred, y_s_distill.detach(), self.T)
            meta_train_loss += self.trade2 * kd_loss

        # mmd loss
        for model_domain in range(nmb):
            for i in range(self.num_domains):
                for j in range(i+1,self.num_domains):
                    penalty += self.mmd(total_all_f_s[model_domain][i], total_all_f_s[model_domain][j])

        meta_train_loss += penalty

        # Dirichlet Mixup
        all_one_hot_labels = torch.cat(all_one_hot_labels, dim=0)

        for model_domain in range(self.num_domains):
            # MixUp
            all_f_s = torch.cat(total_all_f_s[model_domain], dim=0)
            domains = [0] * self.bz
            domains = torch.LongTensor(domains)

            mixup_features = []
            mixup_labels = []
            for i in range(self.num_domains):
                mix_indeces = get_sample_mixup_random(domains)
                mixup_features.append(all_f_s[(i * self.bz) : ((i + 1) * self.bz)][mix_indeces])
                mixup_labels.append(all_one_hot_labels[(i * self.bz) : ((i + 1) * self.bz)][mix_indeces])

            mixup_dir_list = [self.mixup_dir2] * self.num_domains
            mixup_dir_list[model_domain] = self.mixup_dir

            mixup_ratios = get_ratio_mixup_Dirichlet(domains, mixup_dir_list)
            mixup_ratios = mixup_ratios.to(self.device)  # N * 3

            mixup_features = torch.stack(mixup_features)  # 3 * N * D
            mixup_labels = torch.stack(mixup_labels)  # 3 * N * C

            mixup_ratios = mixup_ratios.permute(1, 0).unsqueeze(-1)

            mixup_features = torch.sum((mixup_features * mixup_ratios), dim=0)
            mixup_labels = torch.sum((mixup_labels * mixup_ratios), dim=0)

            # mixup_features_predictions = model.heads[model_domain](mixup_features)
            mixup_features_predictions = self.classifiers[model_domain](mixup_features)
            mixup_feature_loss = manual_CE(mixup_features_predictions, mixup_labels)

            mixup_loss += mixup_feature_loss

        meta_train_loss += self.trade * mixup_loss

        # meta_val_loss = torch.tensor([0], device=self.device)
        total_loss = meta_train_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        return {
            "class": cls_loss.item(),
            "mmd": penalty.item(),
            'distill': kd_loss.item(),
            "loss": total_loss.item(),
        }
    
    def fit(self, dataset):
        """
        Model training.

        input: training dataset
        """
        train_data = CustomDataset(dataset.data, dataset.labels, dataset.domain_labels)
        random_sampler = RandomSampler(train_data, replacement=True)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=random_sampler)

        for epoch in range(self.epoch):
            total_loss = 0.0
            for step in range(self.n_steps_per_epoch):
                batch_data, batch_labels, batch_domains = next(iter(train_loader))
                batch_data, batch_labels, batch_domains = batch_data.to(self.device).permute(0, 3, 1, 2),\
                      batch_labels.to(self.device), batch_domains.to(self.device)

                res = self.update(batch_data, batch_labels, batch_domains)
                
                total_loss += res['loss']

            logging.info(f"Epoch [{epoch+1}/{self.epoch}], Loss: {total_loss:.4f}")

    def get_logits(self, data):
        logits = []
        for i in range(self.num_domains):
            with torch.no_grad():
                inputs = data.to(self.device)
                z = self.featurizers[i](inputs).squeeze(-1).squeeze(-1)
                logits.append(self.classifiers[i](z))

        mean_logits = torch.mean(torch.stack(logits), dim=0)

        return mean_logits
    
    def predict(self, dataset):
        ood_true = np.array(dataset.ood_labels)

        max_logits = []
        y_binaries = []
        logits_list = []
        data = CustomDataset(dataset.data, dataset.labels,dataset.domain_labels)
        dataloader = DataLoader(data, batch_size=256, shuffle=False)
        with torch.no_grad():
            for x,y,_ in dataloader:
                x = torch.tensor(x).permute(0, 3, 1, 2).float()
                y_binary = [True if (c.item() in range(2)) else False for c in y]
                y_binaries.extend(y_binary)
                p = self.get_logits(x)  # p means logits. torch.Size([64, 65])
                logits_list.append(p)
                max_logit: torch.tensor = p.max(dim=1).values
                max_logits.extend(max_logit.tolist())

        mean_logits = torch.mean(torch.stack(logits_list), dim=0)
        ood_score = max_logits

        
        id_logits = mean_logits[:, :2]
        id_predicted_all = torch.argmax(id_logits, dim=1)
        id_indices = np.where(ood_true == 0)[0]
        id_predicted_labels = id_predicted_all[id_indices]
        id_true_labels = np.array(dataset.labels)[id_indices]

        
        return ood_score, ood_true, None, id_predicted_labels, id_true_labels
