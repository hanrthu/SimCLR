import torch
from models.resnet_simclr import ResNetSimCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.margin_triplet import MarginTripletLoss
from loss.nt_logistic import NTLogisticLoss
import os
import shutil
import sys

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.nt_xent_criterion = self._get_loss_strategy(self.device, config['batch_size'], **config['loss'])
 

    def _get_loss_strategy(self,device,batch_size, temperature, use_cosine_similarity,mode,semi_hard='No'):
        if mode == 'nt-xent':
            print('The Training Loss is NT-Xent.')
            return NTXentLoss(device,batch_size, temperature, use_cosine_similarity)
        elif mode == 'nt-logistic':
            print('The Training Loss is NT-Logistic')
            return NTLogisticLoss(device,batch_size, temperature, use_cosine_similarity,semi_hard)
        elif mode == 'margin-triplet':
            print('The Training Loss is MarginTriplet')
            return MarginTripletLoss(device,batch_size, temperature, use_cosine_similarity,semi_hard)
        else:
            print("Unknown mode chosen,using default nt-xent instead.")
            return NTXentLoss(device,batch_size, temperature, use_cosine_similarity)


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _step(self, model, xis, xjs, n_iter):

        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = self.nt_xent_criterion(zis, zjs)
        return loss

    def top_1_step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        correct,total = self.nt_xent_criterion.top_1_eval(zis,zjs)
        return correct,total
    
    def top_5_step(self, model, xis, xjs, n_iter):
        ris, zis = model(xis)  # [N,C]

        # get the representations and the projections
        rjs, zjs = model(xjs)  # [N,C]

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        correct,total = self.nt_xent_criterion.top_5_eval(zis,zjs)
        return correct,total
        
    def test(self):
        train_loader, valid_loader, test_loader= self.dataset.get_data_loaders()
        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)
        self.eval(test_loader,model)

    def train(self):

        train_loader, valid_loader, test_loader= self.dataset.get_data_loaders()

        model = ResNetSimCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            print("Epoch:",end=":")
            print(epoch_counter)
            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

        self.eval(test_loader)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('../runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs), _ in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        print("Valid_loss:",end=":")
        print(valid_loss)
        return valid_loss

    def eval(self, test_loader,model):
        top1 = 0
        top5 = 0
        total = 0
        counter = 0

        with torch.no_grad():
            model.eval()
            for (batch_x, batch_y),_ in test_loader:
                print(batch_x.shape)
                print(batch_y.shape)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                top1 += self.top_1_step(model,batch_x,batch_y,counter)
                top5 += self.top_5_step(model,batch_x,batch_y,counter)
                
                total += 2 * batch_x.size(0)
                counter += 1

        top1_acc = 100 * top1 / total
        top5_acc = 100 * top5 / total
        print("Top1 Accuracy: %d" %(top1_acc))
        print("Top5 Accuracy: %d" %(top5_acc))

        model.train()
        return final_acc
