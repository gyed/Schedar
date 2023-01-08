import numpy as np
import torch


def GPU_info(gpu_list):
    for gpu_id in gpu_list:
        print('GPU{}-{:.4f}G  '.format(gpu_id, torch.cuda.memory_allocated(gpu_id)/1024.**3), end='')
    print('')
        
def write_log(log_path, log_txt):
    with open(log_path, 'a') as f:
        f.write(log_txt)
        f.write("\n")

def print_label_stat(device_id, subset, num_classes):
    temp_dict = {}
    for class_id in range(num_classes):
        temp_dict[class_id] = 0
    for sample in subset:
        label = sample[1]
        temp_dict[label] += 1
    print('D-'+str(device_id), end='\t')
    for class_id in range(num_classes):
        print(temp_dict[class_id], end='\t')
    print(sum(temp_dict.values()))
    
    
def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('model size：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)
