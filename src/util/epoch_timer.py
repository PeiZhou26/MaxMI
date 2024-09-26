"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import numpy as np
import torch

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def scale_data(input_tensor):
    min_val = np.min(input_tensor)
    max_val = np.max(input_tensor)
    scaled_tensor = 2 * (input_tensor - min_val) / (max_val - min_val) - 1
    return scaled_tensor

def mean_variance_loss(tensor, target_mean=0.0, target_variance=1/3):
    
    min_vals = tensor.min(dim=0, keepdim=True)[0]
    max_vals = tensor.max(dim=0, keepdim=True)[0]
    tensor = 2*(tensor - min_vals) / (max_vals - min_vals)-1
    
    means = tensor.mean(dim=0)
    variances = tensor.var(dim=0, unbiased=False)
    
    mean_loss = torch.mean((means - target_mean) ** 2)
    variance_loss = torch.mean((variances - target_variance) ** 2)
    
    total_loss = mean_loss + variance_loss
    return total_loss