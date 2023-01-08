import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import random
import copy
import time
from models.resnet import ResNet, BasicBlock
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
warnings.filterwarnings('ignore')


__all__ = ['FedAvgServer', 'FedAvgClient']


class FedAvgServer():
    def __init__(self, Device_id_list, gpu_id=None, num_classes=10, num_channel=1):
        super(FedAvgServer, self).__init__()
        self.id_list = Device_id_list
        self.client_history_repo = {}
        for client_id in Device_id_list:
            self.client_history_repo[client_id] = []
        self.num_classes = num_classes
        self.soft_label = torch.zeros(1, self.num_classes)
        self.central_model = ResNet(BasicBlock, num_blocks=[1,1,1], num_demensions=[8,8,8], in_channels=num_channel)
        self.teacher_model = copy.deepcopy(self.central_model)
        self.gpu_id = gpu_id
        if self.gpu_id is not None:
            self.central_model.cuda(self.gpu_id)
            self.teacher_model.cuda(self.gpu_id)
        self.optimizer = optim.Adam(self.central_model.parameters(), lr=0.01)
            
    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
    
    def get_delta(self, A, B=None):
        if B == None:
            B = copy.deepcopy(self.central_model.state_dict())
        delta = copy.deepcopy(A)
        for k in delta.keys():
            delta[k] = A[k] - B[k]
        return delta
    
    def sum_delta(self, w):
        w_sum = copy.deepcopy(w[0])
        for k in w_sum.keys():
            for i in range(1, len(w)):
                w_sum[k] += w[i][k]
        return w_sum

    def store_client_historical_update(self, device_dict={}):
        for k, v in device_dict.items():
            self.client_history_repo[k] = self.client_history_repo.get(k, [])
            self.client_history_repo[k].append(self.get_delta(copy.deepcopy(v.client_model.state_dict())))
            if len(self.client_history_repo[k]) == 2:
                self.client_history_repo[k] = [self.sum_delta(self.client_history_repo[k])]
                
                
    def unlearn_client(self, client_id):
        if self.client_history_repo.get(client_id, []) == []:
            print('Client history is empty.')
        else:
            self.teacher_model = copy.deepcopy(self.central_model)
            central_para = self.get_delta(copy.deepcopy(self.central_model.state_dict()),
                                          self.client_history_repo[client_id][0])
            self.central_model.load_state_dict(central_para)
            del self.client_history_repo[client_id]
        
        
    def update_soft_label(self, ref_loader):
        '''Update the soft_label of the main model on the reference dataset.'''
        self.soft_label = torch.zeros(1, self.num_classes)
        if self.gpu_id is not None:
            self.soft_label = self.soft_label.cuda(self.gpu_id)
        self.teacher_model.eval()
        for batch_idx, (data, target) in enumerate(ref_loader):
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.teacher_model(data).detach()
            self.soft_label = torch.cat((self.soft_label, output), 0)
        self.soft_label = self.soft_label[torch.arange(self.soft_label.size(0))!=0]  # Prune the first line 

        
    def remedy_central_model(self, ref_set, training_time=60, batch_size=128, T=5.0, retrain=False): 
        '''The teacher_model supervises the central_model based on the reference dataset.'''
        start_time = time.time()
        self.central_model.train()
        if retrain:
            self.optimizer = optim.Adam(self.central_model.parameters(), lr=0.01)
        imi_info = 'remedy_central_model'
        if self.soft_label.size(0) == 1:
            print('Warning: remedy_central_model before update_soft_label. Operations skiped: ' + imi_info)
        else:
            for iter_idx in range(100*batch_size*training_time):
                idxs = torch.randint(len(ref_set), (batch_size,)) # sample one batch from the reference set
                data, target = [0]*batch_size, [0]*batch_size
                for k, v in enumerate(idxs):
                    data[k], _ = ref_set[v]
                    target[k] = self.soft_label[v]
                data = torch.stack(data, dim = 0)
                target = torch.stack(target, dim = 0)
                if self.gpu_id is not None:
                    data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                self.optimizer.zero_grad()
                output = F.log_softmax(self.central_model(data)/T, dim=1)
                target = F.softmax(target.float()/T, dim=-1)
                loss_neighbor = F.kl_div(output, target.detach(), reduction='batchmean')
                loss_neighbor.backward()
                self.optimizer.step()
                if iter_idx % 50 == 0:
                    print('{} - Epoch: {:3d} \tLoss: {:.6f}'.format(imi_info, iter_idx, loss_neighbor.item()))
                time_gap = time.time() - start_time
                if time_gap >= training_time:
                    print('remedy time consumption(s): {}'.format(time_gap))
                    break
        self.central_model.eval()
        


    def update_central_model(self, frac=1.0, device_dict={}):
        '''update central model on the server. '''
        client_para_list = []
        m = max(int(frac * len(self.id_list)), 1)
        client_idxs = np.random.choice(range(len(self.id_list)), m, replace=False)
        for client_id in client_idxs:
            client_para_list.append(copy.deepcopy(device_dict[client_id].client_model.state_dict()))
        central_para = self.FedAvg(client_para_list)
        self.central_model.load_state_dict(central_para)
        
        
    def distribute_central_model(self, device_dict={}):
        central_para = copy.deepcopy(self.central_model.state_dict())
        for client_id in self.id_list:
            device_dict[client_id].client_model.load_state_dict(central_para)


    def validate_central_model(self, test_loader):
        '''Validate central_model on test dataset. '''
        self.central_model.eval()
        test_loss = 0.
        correct = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        for data, target in test_loader:
            if self.gpu_id is not None:
                data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
            output = self.central_model(data)
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


class FedAvgClient():
    def __init__(self, Device_id, gpu_id=None, num_classes=10, num_channel=1):
        super(FedAvgClient, self).__init__()
        self.id = Device_id
        self.num_classes = num_classes
        self.client_model = ResNet(BasicBlock, num_blocks=[1,1,1], num_demensions=[8,8,8], in_channels=num_channel)
        self.gpu_id = gpu_id
        self.optimizer = optim.Adam(self.client_model.parameters(), lr=0.01)
        if self.gpu_id is not None:
            self.client_model.cuda(self.gpu_id)
        
    def update_client_model(self, num_iter, local_loader=None):
        self.client_model.train()
       
        for iter_idx in range(num_iter):
            for batch_idx, (data, target) in enumerate(local_loader):
                self.optimizer.zero_grad()
                if self.gpu_id is not None:
                    data, target = data.cuda(self.gpu_id), target.cuda(self.gpu_id)
                output = self.client_model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
            if iter_idx % 10 == 0:
                print('Client{:2d}\tEpoch:{:3d}\t\tLoss: {:.8f}'.format(self.id, iter_idx, loss.item()))