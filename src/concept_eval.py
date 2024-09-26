import os, sys
import pickle as pkl
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from model import KFE_m
import torch.nn.functional as F
from concept_train import traj_obs, IndexedDataset
import torch
from infer_manisk_nond import func_mi, model_mi
from path import PKL_PATH, MODEL_PATH

def batch_linear_interpolate2(tensor, indices, shift=-1):
    a, b, c = tensor.shape
    indices_floor = torch.floor(indices).long()+shift
    indices_ceil = indices_floor + 1
    weights = indices - indices_floor.float()

    indices_ceil = torch.clamp(indices_ceil, 0, b-1)
    indices_floor = torch.clamp(indices_floor, 0, b-1)

    values_floor = tensor[torch.arange(a), indices_floor]
    values_ceil = tensor[torch.arange(a), indices_ceil]
    interpolated_values = (1 - weights.unsqueeze(1)) * values_floor + weights.unsqueeze(1) * values_ceil
    
    return interpolated_values

if __name__=="__main__":
    task_list = ["PegInsertionSide-v0"] # can add other tasks
    task_name = task_list[0]
    data, _ = traj_obs(task_name)
    
    checkpoint_path = f"{MODEL_PATH}/{task_name}"
    checkpoint = torch.load(checkpoint_path)
    
    data_list = data["obs"]
    data_list = [torch.from_numpy(arr) for arr in data_list]
    traj_len_list = [len(arr) for arr in data_list]
    min_len, max_len = min(traj_len_list), max(traj_len_list)
    
    data_list2 = []
    for tensor in data_list:
        tensor_b = F.interpolate(tensor.unsqueeze(dim=0).transpose(1,2), size=min_len, mode='linear').transpose(2,1).squeeze()
        data_list2.append(tensor_b)
    batch = torch.stack(data_list2).cuda()

    num_concept = 5
    embed_size = 10
    
    batch_size = len(batch)
    time_range = torch.arange(min_len).cuda()
    concept_idx = torch.arange(num_concept).repeat_interleave(batch_size).cuda()
    
    dataset = IndexedDataset(batch)
    model = KFE_m(input_dim=dataset.dim+2*embed_size, output_dim=min_len, traj_len=min_len, text_len=num_concept, embed_size=embed_size).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval()
    with torch.no_grad():
        samples = batch.cuda()
        samples = samples.repeat(num_concept, 1, 1).float()
        dist = model(samples, time_range, concept_idx) # [B, min_len] -> [B*N, min_len]
        prob = F.softmax(dist,dim=-1)
        weighted_range = torch.sum(torch.mul(prob,time_range.unsqueeze(0)),dim=1)
        weighted_range = weighted_range.view(num_concept,-1)        
        keyidx_dict = {f"traj_{i}": torch.round(weighted_range[:, i]*traj_len_list[i]/min_len).int().detach().cpu() for i in range(len(batch))}
        
        keyidx_path = f"{PKL_PATH}/{task_name}/concepts.pkl"
        os.makedirs(os.path.dirname(keyidx_path), exist_ok=True)
        with open(keyidx_path, "wb") as f:
            pkl.dump(keyidx_dict, f)  