import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import random
import copy
from models.resnet import ResNet, BasicBlock
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')


__all__ = ['DSGD_Device']

class DSGD_Device():
    def __init__(self, Device_id, gpu_id=None, num_classes=10, num_channel=1):
        super(DSGD_Device, self).__init__()
        self.id = Device_id
        self.num_classes = num_classes
        self.model = ResNet(BasicBlock, num_blocks=[1,1,1], num_demensions=[8,8,8], in_channels=num_channel)
        self.gpu_id = gpu_id
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.neighbor_list = []
        if self.gpu_id is not None:
            self.model.cuda(self.gpu_id)
        
    def train_local_model(self, num_iter, local_loader=None):
        self.model.train()
        for iter_idx in range(num_iter):
            for batch_idx, (data, target) in enumerate(local_loader):
                self.optimizer.zero_grad()
                if self.gpu_id is not None:
                    data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
            if iter_idx % 10 == 0:
                print('Client{:2d}\tEpoch:{:3d}\t\tLoss: {:.8f}'.format(self.id, iter_idx, loss.item()))
                
    def Neighbor_Avg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
    
    def Merging(self, knowledge, gamma):
        w_mer = copy.deepcopy(self.model.state_dict())
        for k in w_mer.keys():
            w_mer[k] = w_mer[k].float()
            w_mer[k] += knowledge[k]*torch.tensor(gamma)
            w_mer[k] = torch.div(w_mer[k], torch.tensor(1+gamma))
        return w_mer
    
    def communication(self, frac=1.0, gamma=0.1, device_dict={}):
        '''update central model on the server. '''
        neighbor_para_list = []
        m = max(int(frac * len(self.neighbor_list)), 1)
        neighbor_idxs = np.random.choice(range(len(self.neighbor_list)), m, replace=False)
        
        for neighbor_id in neighbor_idxs:
            neighbor_para_list.append(copy.deepcopy(device_dict[neighbor_id].model.state_dict()))
        avg_neighbor_para = self.Neighbor_Avg(neighbor_para_list)
        updated_model_para = self.Merging(avg_neighbor_para, gamma=gamma)
        self.model.load_state_dict(updated_model_para)
        
    def validate_local_model(self, test_loader):
        '''Validate main model on test dataset. '''
        self.model.eval()
        test_loss = 0.
        correct = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        for data, target in test_loader:
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            batch_len = len(target.cpu().data)
            precision += precision_score(target.cpu().data, pred.cpu(), average='macro')*batch_len
            recall += recall_score(target.cpu().data, pred.cpu(), average='macro')*batch_len
            f1 += f1_score(target.cpu().data, pred.cpu(), average='macro')*batch_len
        len_ = len(test_loader.dataset)
        test_loss /= len_
        print('Server - Avg_loss: {:.4f}, Acc: {}/{} ({:.4f})'.format(test_loss, correct,len_,correct/len_))      
        return correct/len_, precision/len_, recall/len_, f1/len_