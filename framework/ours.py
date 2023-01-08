import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import random
from models.resnet import ResNet, BasicBlock
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')

__all__ = ['Device']

class Device():
    def __init__(self, Device_id, gpu_id=None, num_classes=10, num_channel=1):
        super(Device, self).__init__()
        self.id = Device_id
        self.num_classes = num_classes
        self.main_model = ResNet(BasicBlock, num_blocks=[8,8,8], in_channels=num_channel)
        self.seed_model = ResNet(BasicBlock, num_blocks=[3,3,3], num_demensions=[32,32,32], in_channels=num_channel)
        self.soft_label = torch.zeros(1, self.num_classes)
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.main_model.cuda(self.gpu_id)
            self.seed_model.cuda(self.gpu_id)
        self.neighbor_list = []
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=0.01)

    def initialize_seed_model(self, ref_loader):
        self.seed_model = models.__dict__[seed_model_type](num_classes=self.num_classes)


    def train_main_model(self, num_iter, local_loader):
        '''Train main model on the local dataset. '''
        self.main_model.train()
        print('\nDevice: '+ str(self.id) + ' main model training')
        for iter_idx in range(num_iter):
            for batch_idx, (data, target) in enumerate(local_loader):
                if self.gpu_id is not None:
                    data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                output = self.main_model(data)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
            if iter_idx % 10 == 0:
                print('Epoch:{:3d}\t\tLoss: {:.8f}'.format(iter_idx, loss.item()))


    def update_soft_label(self, ref_loader):
        '''Update the soft_label of the main model on the reference dataset.'''
        self.soft_label = torch.zeros(1, self.num_classes)
        if self.gpu_id is not None:
            self.soft_label = self.soft_label.cuda(self.gpu_id)
        self.main_model.eval()
        for batch_idx, (data, target) in enumerate(ref_loader):
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.main_model(data).detach()
            self.soft_label = torch.cat((self.soft_label, output), 0)
        self.soft_label = self.soft_label[torch.arange(self.soft_label.size(0))!=0]  # Prune the first line 


    def train_seed_model(self, ref_set, num_iter, batch_size=128, T=3.0): 
        '''The main model supervises the seed model based on the reference dataset.'''
        self.seed_model.train()
        optimizer_ = optim.Adam(self.seed_model.parameters(), lr=0.01)
        imi_info = 'Device:{} Trn_Seed'.format(self.id)
        if self.soft_label.size(0) == 1:
            print('Warning: train_seed_model before update_soft_label. Operations skiped: ' + imi_info)
        else:
            for iter_idx in range(num_iter):
                idxs = torch.randint(len(ref_set), (batch_size,)) # sample one batch from the reference set
                data, target = [0]*batch_size, [0]*batch_size
                for k, v in enumerate(idxs):
                    data[k], _ = ref_set[v]
                    target[k] = self.soft_label[v]
                data = torch.stack(data, dim = 0)
                target = torch.stack(target, dim = 0)
                if self.gpu_id is not None:
                    data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                optimizer_.zero_grad()
                output = F.log_softmax(self.seed_model(data)/T, dim=1)
                target = F.softmax(target.float()/T, dim=-1)
                loss_neighbor = F.kl_div(output, target.detach(), reduction='batchmean')
                loss_neighbor.backward()
                optimizer_.step()
                if iter_idx % 50 == 0:
                    print('{} - Epoch: {:3d} \tLoss: {:.6f}'.format(imi_info, iter_idx, loss_neighbor.item()))
        self.seed_model.eval()


    def validate_main_model(self, test_loader):
        '''Validate main model on test dataset. '''
        self.main_model.eval()
        test_loss = 0.
        correct = 0.
        accuracy = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        for data, target in test_loader:
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.main_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            precision += precision_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            recall += recall_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            f1 += f1_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
        
        len_ = len(test_loader.dataset)
        test_loss /= len_
        print('Device:{:2d} Val_main - Avg_loss: {:.4f}, Acc: {}/{} ({:.4f})'.format(self.id, test_loss, correct,
                                                                                  len_, correct/len_))
        return correct/len_, precision/len_, recall/len_, f1/len_


    def seed_model_predition(self, data):
        self.seed_model.eval()
        if self.gpu_id is not None:
            data = data.cuda(self.gpu_id)
        return self.seed_model(data).detach()


    def validate_ensamble(self, test_loader, device_dict, rho=0.5):
        '''Validate the ensamble of the main model and sub_models. '''
        self.main_model.eval()
        correct = 0.
        accuracy = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        for data, target in test_loader:
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output_local = self.main_model(data)
            output_neighbor = torch.zeros_like(output_local)
            if rho != 0:
                for neighbor_id in self.neighbor_list:
                    # Identical to generate predictions on a local copy (sub_model) of the neighbor's seed_model.
                    temp = device_dict[neighbor_id].seed_model_predition(data)
                    if self.gpu_id is not None:
                        temp = temp.cuda(self.gpu_id)
                    output_neighbor += temp
            if len(self.neighbor_list) == 0:
                rho = 0
            else:
                output_neighbor /= len(self.neighbor_list)
            output_combined = output_local * (1-rho) + output_neighbor * rho
            pred = output_combined.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            precision += precision_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            recall += recall_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
            f1 += f1_score(target.cpu().data, pred.cpu(), average='macro')*len(target.cpu().data)
        len_ = len(test_loader.dataset)
        print('Device: {:2d} Val_ensamble - Acc: {}/{} ({:.4f})'.format(self.id, correct, len_, correct/len_))
        return correct/len_, precision/len_, recall/len_, f1/len_



    def unlearn_neighbor(self, neighbor_id=None, neighbor_num=None):
        if neighbor_id is not None:
            del self.neighbor_list[neighbor_id]
        if neighbor_num is not None: 
            self.neighbor_list = self.neighbor_list[:-n]


    def add_neighbor(self, neighbor_id=None):
        self.neighbor_list.append(neighbor_id)