import numpy as np
import logging
import torch
import torch.nn as nn
import random
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

class MEDIC:
    def __init__(self, epochs =1000, batch_size = 64,num_classes=2, threshold_percent=90,
                 lr=0.0001, meta_lr=0.01):
        """
        MEDIC
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percent = threshold_percent
        self.num_classes = num_classes
        self.meta_lr = meta_lr
        self.energy_threshold = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MutiClassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
        self.net = self.net.to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.ovaloss = OVALoss()

    def compute_energy(self, logits, T=1.0):
        # Compute energy using the logits
        return -T * torch.logsumexp(logits / T, dim=1)

    def fit(self, dataset):
        """
        Train the One-Class SVM model using ID data only.
        :param dataset: A dataset object with dataset.data (images) and dataset.ood_labels (0 for ID, 1 for OOD)
        """
        source_domain = np.sort(np.unique(dataset.domain))
        known_classes = [0,1]
        unknown_classes = [2]
        num_domain = len(source_domain)
        num_classes = len(known_classes)

        class_index = [i for i in range(num_classes)]
        group_length = (num_classes-1) // 10 + 1
        group_index = [i for i in range((num_classes-1)//group_length + 1)]
        num_group = len(group_index)

        images = torch.tensor(dataset.data).permute(0, 3, 1, 2).float()

        domain_specific_loader = []
        for domain in source_domain:       
            dataloader_list = []
            domain_indices = np.where(dataset.domain == domain)[0].tolist()
            domain_data = images[domain_indices]
            domain_labels = dataset.labels[domain_indices]

            for i, classes in enumerate(known_classes):
                cls_indices = np.where(domain_labels == classes)[0].tolist()
                cls_data = images[cls_indices]
                cls_labels = domain_labels[cls_indices]
                # if len(cls_labels > 0):
                #     scd = CustomDataset(images, dataset.labels)
                # else:
                #     scd = CustomDataset(domain_data, domain_labels)
                scd = CustomDataset(images, dataset.labels)
                loader = DataLoader(dataset=scd, batch_size=self.batch_size//2, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)


            domain_specific_loader.append(ConnectedDataIterator(dataloader_list=dataloader_list, batch_size=self.batch_size))


        exp_domain_index = 0   
        exp_group_num = (num_group-1) // 3 + 1
        exp_group_index = random.sample(group_index, exp_group_num)

        domain_index_list = [i for i in range(num_domain)]

        fast_parameters = list(self.net.parameters())
        for weight in self.net.parameters():
            weight.fast = None
        self.net.zero_grad()

        energies = []
        for epoch in range(self.epochs):
            #################################################################### meta train open

            self.net.train()
            meta_train_loss = meta_val_loss = 0

            domain_index_set = set(domain_index_list) - {exp_domain_index}
            i, j = random.sample(list(domain_index_set), 2)

            domain_specific_loader[i].remove(exp_group_index)
            input, label = next(domain_specific_loader[i])      
            domain_specific_loader[i].reset()  

            input = input.to(self.device)
            label = label.to(self.device)
            out, output = self.net.c_forward(x=input)
            meta_train_loss += self.criterion(out, label)
            output = output.view(output.size(0), 2, -1)
            meta_train_loss += self.ovaloss(output, label)

            domain_specific_loader[j].remove(exp_group_index)
            input, label = next(domain_specific_loader[j])
            domain_specific_loader[j].reset()

            input = input.to(self.device)
            label = label.to(self.device)
            out, output = self.net.c_forward(x=input)
            meta_train_loss += self.criterion(out, label)
            output = output.view(output.size(0), 2, -1)
            meta_train_loss += self.ovaloss(output, label)

            domain_specific_loader[exp_domain_index].keep(exp_group_index)
            input, label = next(domain_specific_loader[exp_domain_index])
            domain_specific_loader[exp_domain_index].reset()

            input = input.to(self.device)
            label = label.to(self.device)
            out, output = self.net.c_forward(x=input)
            energy = self.compute_energy(out)
            energies.append(energy)
            meta_train_loss += self.criterion(out, label)
            output = output.view(output.size(0), 2, -1)
            meta_train_loss += self.ovaloss(output, label)

    ########################################################################## meta val open

            grad = torch.autograd.grad(meta_train_loss, fast_parameters,
                                    create_graph=True, allow_unused=True)

            for k, weight in enumerate(self.net.parameters()):
                if grad[k] is not None:
                    if weight.fast is None:
                        weight.fast = weight - self.meta_lr * grad[k]
                    else:
                        weight.fast = weight.fast - self.meta_lr * grad[
                            k]

            domain_specific_loader[i].keep(exp_group_index)
            input_1, label_1 = domain_specific_loader[i].next(batch_size=self.batch_size//2)      
            domain_specific_loader[i].reset() 

            domain_specific_loader[j].keep(exp_group_index)
            input_2, label_2 = domain_specific_loader[j].next(batch_size=self.batch_size//2)      
            domain_specific_loader[j].reset() 
            
            input = torch.cat([input_1, input_2], dim=0)
            label = torch.cat([label_1, label_2], dim=0)

            input = input.to(self.device)
            label = label.to(self.device)
            out, output = self.net.c_forward(x=input)
            meta_val_loss += self.criterion(out, label)
            output = output.view(output.size(0), 2, -1)
            meta_val_loss += self.ovaloss(output, label)

            for i in range(2):

                domain_specific_loader[exp_domain_index].remove(exp_group_index)
                input, label = next(domain_specific_loader[exp_domain_index])
                domain_specific_loader[exp_domain_index].reset()

                input = input.to(self.device)
                label = label.to(self.device)
                out, output = self.net.c_forward(x=input)
                meta_val_loss += self.criterion(out, label)
                output = output.view(output.size(0), 2, -1)
                meta_val_loss += self.ovaloss(output, label)

    ##################################################################### 

            total_loss = meta_train_loss + meta_val_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            fast_parameters = list(self.net.parameters())
            for weight in self.net.parameters():
                weight.fast = None
            self.net.zero_grad()
        
        energies = torch.cat(energies, dim=0).detach().numpy()
        self.energy_threshold = np.percentile(energies, self.threshold_percent)
    
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
        self.net.eval()
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                logits, _ = self.net.c_forward(x=images)
                logit_list.append(logits.cpu())
                energy = self.compute_energy(logits)
                energy_list.append(energy.cpu())
        test_energy = torch.cat(energy_list, dim=0).detach().numpy()
        
        energy_predicted_labels = (test_energy > self.energy_threshold).astype(int)
        ood_true = np.array(dataset.ood_labels)
        
        id_indices = np.where(energy_predicted_labels == 0)[0]
        id_true_labels = np.array(dataset.labels)[id_indices]
        
        logit_list = torch.cat(logit_list, dim=0).cpu().numpy()
        id_logits = logit_list[id_indices]
        
        id_predicted = np.argmax(id_logits, axis=1)
        id_predicted_labels = id_predicted

        return test_energy, ood_true, energy_predicted_labels, id_true_labels, id_predicted_labels

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class ConnectedDataIterator:
    def __init__(self, dataloader_list, batch_size):
        self.dataloader_list = dataloader_list
        self.batch_size = batch_size
        
        self.length = len(self.dataloader_list)
        self.iter_list = [iter(loader) for loader in self.dataloader_list]
        self.available_set = set([i for i in range(self.length)])

    def append(self, index_list):
        self.available_set = self.available_set | set(index_list)

    def keep(self, index_list):
        self.available_set = set(index_list)

    def remove(self, index_list):
        self.available_set = self.available_set - set(index_list)

    def reset(self):
        self.available_set = set([i for i in range(len(self.dataloader_list))])

    def __next__(self):
        data_sum = []
        label_sum = []
        for i in self.available_set:
            try:
                data, label, *_ = next(self.iter_list[i])
            except StopIteration:
                self.iter_list[i] = iter(self.dataloader_list[i])
                data, label, *_ = next(self.iter_list[i])
            data_sum.append(data)
            label_sum.append(label)
        
        data_sum = torch.cat(data_sum, dim=0)
        label_sum = torch.cat(label_sum, dim=0)
        
        rand_index = random.sample([i for i in range(len(data_sum))], min(self.batch_size, data_sum.shape[0]))

        return data_sum[rand_index], label_sum[rand_index]

    def next(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        data_sum = []
        label_sum = []
        for i in self.available_set:
            try:
                data, label, *_ = next(self.iter_list[i])
            except StopIteration:
                self.iter_list[i] = iter(self.dataloader_list[i])
                data, label, *_ = next(self.iter_list[i])
            data_sum.append(data)
            label_sum.append(label)
        
        data_sum = torch.cat(data_sum, dim=0)
        label_sum = torch.cat(label_sum, dim=0)
        
        rand_index = random.sample([i for i in range(len(data_sum))], batch_size)

        return data_sum[rand_index], label_sum[rand_index]
    
class OVALoss(nn.Module):
    def __init__(self):
        super(OVALoss, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, label):
        assert len(input.size()) == 3
        assert input.size(1) == 2

        input = F.softmax(input, 1)
        label_p = torch.zeros((input.size(0),
                           input.size(2))).long().to(self.device)
        label_range = torch.range(0, input.size(0) - 1).long()
        label_p[label_range, label] = 1
        label_n = 1 - label_p
        open_loss_pos = torch.mean(torch.sum(-torch.log(input[:, 1, :]
                                                    + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(input[:, 0, :] +
                                                1e-8) * label_n, 1)[0])
        return 0.5*(open_loss_pos + open_loss_neg)

        
class MutiClassifier(nn.Module):
    def __init__(self, net, num_classes, feature_dim=512):
        super(MutiClassifier, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.classifier = Linear_fw(feature_dim, self.num_classes)
        self.b_classifier = Linear_fw(feature_dim, self.num_classes*2)
        nn.init.xavier_uniform_(self.classifier.weight, .1)
        nn.init.constant_(self.classifier.bias, 0.)
        nn.init.xavier_uniform_(self.b_classifier.weight, .1)
        nn.init.constant_(self.b_classifier.bias, 0.)

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x

    def b_forward(self, x):
        x = self.net(x)
        x = self.b_classifier(x)
        return x

    def c_forward(self, x):
        x = self.net(x)
        x1 = self.classifier(x)
        x2 = self.b_classifier(x)
        return x1, x2
    
def resnet50_fast(progress=True):
    model = ResNetFast(Bottleneck, [3, 4, 6, 3])
    # state_dict = load_state_dict_from_url(model_urls['resnet50'],
    #                                       progress=progress)
    # model.load_state_dict(state_dict, strict=False)
    del model.fc

    return model

class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)

    def forward(self, input):
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        """ Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        """Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        if self.weight.fast is not None and self.bias.fast is not None:
            return F.batch_norm(
            input,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight.fast, self.bias.fast, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_fw(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = BatchNorm2d_fw(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_fw(in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d_fw(planes)
        self.conv3 = Conv2d_fw(in_channels=planes, out_channels=planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = BatchNorm2d_fw(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFast(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNetFast, self).__init__()
        self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d_fw(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d_fw(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_fw(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x