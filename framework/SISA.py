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

__all__ = ['SISA_Device']

class SISA_Device():
    def __init__(self, Device_id, gpu_id=None, num_classes=10, num_channel=1):
        super(SISA_Device, self).__init__()
        self.id = Device_id
        self.num_classes = num_classes
        self.main_model = ResNet(BasicBlock, num_blocks=[8,8,8], in_channels=num_channel)
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.main_model.cuda(self.gpu_id)
            self.optimizer = optim.Adam(self.main_model.parameters(), lr=0.01)
        self.neighbor_list = []



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
            if iter_idx % 1 == 0:
                print('Iter: {:3d}\t\tLoss: {:.8f}'.format(iter_idx, loss.item()))
        self.main_model.eval()

    def validate_main_model(self, test_loader):
        '''Validate main model on test dataset. '''
        self.main_model.eval()
        test_loss = 0.
        correct = 0.
        for data, target in test_loader:
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.main_model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('Device:{:2d} Val_main - Avg_loss: {:.4f}, Acc: {}/{} ({:.4f})'.format(self.id, test_loss, correct,
                                                                                  len(test_loader.dataset),
                                                                                  correct / len(test_loader.dataset)))      
        return correct / float(len(test_loader.dataset))
    

    
    def validate_ensamble(self, test_loader, device_dict, compute_gpu_id=0):
        '''
        SISA baseline: An aggregation of all devicess' local main models
        *Conduct this function after Device.local_model_train() for all devices!
        '''
        correct = 0.
        accuracy = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        warning_sign = False
        for data, target in test_loader:
            data, target = data.cuda(self.gpu_id), target.cuda(compute_gpu_id)
            output_SISA = self.main_model(data).cuda(compute_gpu_id)
            
            if len(self.neighbor_list) == 0:
                warning_sign = True
            else:
                for device_id in self.neighbor_list:
                    gpu_id = device_dict[device_id].gpu_id
                    data = data.cuda(gpu_id)
                    output_SISA += device_dict[device_id].main_model(data).cuda(compute_gpu_id).detach()

            pred = output_SISA.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            batch_len = len(target.cpu().data)
            precision += precision_score(target.cpu().data, pred.cpu(), average='macro')*batch_len
            recall += recall_score(target.cpu().data, pred.cpu(), average='macro')*batch_len
            f1 += f1_score(target.cpu().data, pred.cpu(), average='macro')*batch_len
        if warning_sign:
            print('Client {}: Neighor_num = 0'.format(self.id))
        len_ = len(test_loader.dataset)
        print('SISA: Client {} Test -  Accuracy: {}/{} ({:.4f})'.format(self.id, correct, len_,  correct/len_)) 
        return correct/len_, precision/len_, recall/len_, f1/len_ 
        
        
 