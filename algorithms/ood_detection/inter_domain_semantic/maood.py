import os.path
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

class MAOOD_Base(ERM):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MAOOD_Base, self).__init__(input_shape, num_classes, num_domains, hparams)
    
        self.G = load_munit_model(
            self.hparams['mbdg_model_path'],
            self.hparams['mbdg_config_path'])

    def predict(self, x):
        return self.network(x)

    @torch.no_grad()
    def generate_images(self, images):
        delta = torch.randn(images.size(0), self.G.delta_dim, 1, 1).to(images.device).requires_grad_(False)
        return self.G(images, delta)

    @staticmethod
    def calc_feature_dist_reg_l2(self, x, feature):
        mb_images = self.generate_images(x)
        mb_output = self.featurizer(mb_images)
        diff = feature - mb_output
        return torch.norm(diff, dim=1).sum()/feature.size(0)

    def plot_mixup_inds(self, x, x_pos, lambda_values):
        delta = torch.randn(x.size(0), self.G.delta_dim, 1, 1).to(x.device).requires_grad_(False)
        images_to_plot = [x, x]
        for lambda_value in lambda_values:
            images_to_plot.append(self.G.mnist_mixup_ind(x, x, delta, [lambda_value]))

        title_list = ['x1', 'x2'] + [str(l) for l in lambda_values]

        def plot_mixup_images(image_list, title_list, save_dir="domainbed/mixup_imgs"):
            os.makedirs(save_dir, exist_ok=True)
            # Save each image in the batch
            for i in range(image_list[0].size(0)):
                # Create a new figure with three subplots
                fig, axs = plt.subplots(1, len(image_list), figsize=(20, 3))

                for j, x in enumerate(image_list):
                    new_x = torch.zeros(3, 32, 32)
                    new_x[:2, :, :] = x[i]

                    axs[j].imshow(new_x.permute(1, 2, 0).mul(255).byte().numpy())
                    axs[j].set_title(title_list[j])

                # Hide axis ticks and labels
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Show the figure
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{random.randint(0, 100000000)}.png"))

        plot_mixup_images(images_to_plot, title_list, os.path.join("domainbed/mixup_imgs", "ind"))

    # for MNIST
    def plot_mixup_oods(self, x, x_neg_1, gmm, thr, num_classes):
        delta = torch.randn(x.size(0), self.G.delta_dim, 1, 1).to(x.device).requires_grad_(False)
        images_to_plot = [x, x_neg_1]
        images_to_plot.append(self.G.mnist_mixup_ood(x, x_neg_1, delta, gmm, thr, num_classes))
        title_list = ['x1', 'x2', 'ood']

        def plot_mixup_images(image_list, title_list, save_dir="domainbed/mixup_imgs"):
            os.makedirs(save_dir, exist_ok=True)
            # Save each image in the batch
            for i in range(image_list[0].size(0)):
                # Create a new figure with three subplots
                fig, axs = plt.subplots(1, len(image_list), figsize=(10, 3))

                for j, x in enumerate(image_list):
                    new_x = torch.zeros(3, 32, 32)
                    new_x[:2, :, :] = x[i]

                    axs[j].imshow(new_x.permute(1, 2, 0).mul(255).byte().numpy())
                    axs[j].set_title(title_list[j])

                # Hide axis ticks and labels
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Show the figure
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{random.randint(0, 100000000)}.png"))
                plt.close()

        plot_mixup_images(images_to_plot, title_list, os.path.join("domainbed/mixup_imgs", "ood"))

    def generate_synthetic_ood(self, x, x_pos, x_neg_1, x_neg_2, x_neg_3):
        delta = torch.randn(x.size(0), self.G.delta_dim, 1, 1).to(x.device).requires_grad_(False)
        return self.G.mnist_mixup_ood(x, x_neg_1, x_neg_2, x_neg_3, delta)

    @staticmethod
    def relu(x):
        return x if x > 0 else torch.tensor(0).to(x.device)
    
class MAOOD(MAOOD_Base):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MAOOD, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.dual_var = torch.tensor(1.0).cuda().requires_grad_(False)
        self.dual_var_2 = torch.tensor(0.05).cuda().requires_grad_(False)
        self.dual_var_3 = torch.tensor(0.01).cuda().requires_grad_(False)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])

        self.dual_var_2 = self.dual_var_2.cpu().to(all_x.device).requires_grad_(False)

        clean_output = self.predict(all_x)
        clean_loss = F.cross_entropy(clean_output, all_y)

        dist_reg_2 = self.calc_feature_dist_reg_l2(all_x, self.featurizer(all_x))

        loss = clean_loss + self.dual_var_2 * dist_reg_2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        const_unsat_2 = dist_reg_2.detach() - self.hparams['gamma_2']
        self.dual_var_2 = self.relu(self.dual_var_2 + self.hparams['dual_step_size_2'] * const_unsat_2)

        return {'loss': loss.item(), 'dist_reg_2': dist_reg_2.item(), 'dual_var_2': self.dual_var_2.item()}

    def finetune(self, minibatches, unlabeled=None, synthetic_ood=False, gmm=None, thr=0.01, print_energy=False, num_classes=2, ood_classes=None, n_oods=1):

        all_x = torch.cat([x for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])
        all_y = torch.cat([y for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])
        all_x_neg_1 = torch.cat([x_neg_1 for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])

        self.dual_var_2 = self.dual_var_2.cpu().to(all_x.device).requires_grad_(False)

        clean_output = self.predict(all_x)
        clean_loss = F.cross_entropy(clean_output, all_y)

        ind_logits = self.featurizer(all_x)
        dist_reg_2 = self.calc_feature_dist_reg_l2(all_x, ind_logits)

        if synthetic_ood:
            delta = torch.randn(all_x.size(0), self.G.delta_dim, 1, 1).to(all_x.device)
            oods = self.G.mnist_mixup_ood(all_x, all_x_neg_1, delta, gmm, thr, num_classes, n_oods)
            ind_scores = -torch.logsumexp(ind_logits, dim=1)
            ood_logits = self.featurizer(oods)
            ood_scores = -torch.logsumexp(ood_logits, dim=1)

            if print_energy:
                print(f"ind_scores: {ind_scores}")
                print(f"ood_scores: {ood_scores}")

            m_in = -8
            m_out = -6

            ood_reg = torch.pow(F.relu(ind_scores-m_in), 2).mean() + torch.pow(F.relu(m_out-ood_scores), 2).mean()

            loss = clean_loss + self.dual_var_2 * dist_reg_2 + self.dual_var_3 * ood_reg
        else:
            loss = clean_loss + self.dual_var_2 * dist_reg_2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        const_unsat_2 = dist_reg_2.detach() - self.hparams['gamma_2']
        self.dual_var_2 = self.relu(self.dual_var_2 + self.hparams['dual_step_size_2'] * const_unsat_2)
        const_unsat_3 = ood_reg.detach() - self.hparams['gamma_3']
        self.dual_var_3 = self.relu(self.dual_var_3 + self.hparams['dual_step_size_3'] * const_unsat_3)

        return {'cross-entropy': clean_loss.item(), 'dual_var_2': self.dual_var_2.item(), 'dist_reg_2': dist_reg_2.item(), 'dual_var_3': self.dual_var_3.item(), 'ood reg': ood_reg.item(), 'loss': loss.item()}

    # mixup plots only
    def save_images(self, minibatches, lambda_values=None, gmm=None, thr=0.05, num_classes=2):
        all_x = torch.cat([x for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])
        all_x_neg_1 = torch.cat([x_neg_1 for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])

        self.plot_mixup_oods(all_x, all_x_neg_1, gmm, thr, num_classes)

    def save_munit_images(self, minibatches, output_dir):
        all_x = torch.cat([x for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])
        all_y = torch.cat([y for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])

        images_to_plot = [all_x]

        x_reconstruct, x_random_style = self.G.gen_munit_images(all_x)

        images_to_plot.append(x_reconstruct)
        for x_rand in x_random_style:
            images_to_plot.append(x_rand)

        def plot_mixup_images(image_list, labels, save_dir):
            # Save each image in the batch
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            for i in range(image_list[0].size(0)):
                image_dir = os.path.join(save_dir, str(i))
                os.makedirs(image_dir, exist_ok=True)

                for j, x in enumerate(image_list):
                    plt.figure()
                    image = x[i].cpu().numpy().transpose((1, 2, 0))
                    image = std * image + mean
                    image = np.clip(image, 0, 1)
                    plt.imshow(image)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(os.path.join(image_dir, f"domain-{i % 16}-label-{labels[i]}-{j}.png"))
                    plt.close()

        plot_mixup_images(images_to_plot, all_y, output_dir)

    # get semantic vectors
    def save_semantic(self, minibatches, ood_classes=None):
        all_x = torch.cat([x for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])
        all_y = torch.cat([y for x,y, x_pos, x_neg_1, x_neg_2, x_neg_3 in minibatches])

        semantics = self.G.get_semantic(all_x)
        return semantics, all_y