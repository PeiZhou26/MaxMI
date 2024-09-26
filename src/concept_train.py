import os
import pickle as pkl
import random
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import KFE_m

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
from train_utils import CosineAnnealingLRWarmup
from infer_manisk_nond import func_mi, model_mi
from path import DATA_PATH, MODEL_PATH
import h5py

def get_key_states(task, data, idx):
    # Note that `infos` is for the next obs rather than the current obs.
    # Thus, we need to offset the `step_idx`` by one.
    key_states = []
    key_idx_list = []

    # If TurnFaucet (two key states)
    # key state I: is_contacted -> true
    # key state II: end of the trajectory
    if task == 'TurnFaucet-v0':
        for step_idx, key in enumerate(data['infos/is_contacted'][idx]):
            if key: break
        key_idx_list.append(step_idx+1)
        key_states.append(data['obs'][idx][step_idx+1].astype(np.float32))

    # If PegInsertion (three key states)
    # key state I: is_grasped -> true
    # key state II: pre_inserted -> true
    # key state III: end of the trajectory
    if task == 'PegInsertionSide-v0':
        for step_idx, key in enumerate(data['infos/is_grasped'][idx]):
            if key: break
        key_idx_list.append(step_idx+1)
        key_states.append(data['obs'][idx][step_idx+1].astype(np.float32))
        for step_idx, key in enumerate(data['infos/pre_inserted'][idx]):
            if key: break
        key_idx_list.append(step_idx+1)
        key_states.append(data['obs'][idx][step_idx+1].astype(np.float32))
    
    # If PickCube (two key states)
    # key state I: is_grasped -> true
    # key state II: end of the trajectory
    if task == 'PickCube-v0':
        for step_idx, key in enumerate(data['infos/is_grasped'][idx]):
            if key: break
        key_idx_list.append(step_idx+1)
        key_states.append(data['obs'][idx][step_idx+1].astype(np.float32))
    
    # If StackCube (three key states)
    # key state I: is_cubaA_grasped -> true
    # key state II: the last state of is_cubeA_on_cubeB -> true 
    #               right before is_cubaA_grasped -> false
    # key state III: end of the trajectory
    if task == 'StackCube-v0':
        for step_idx, key in enumerate(data['infos/is_cubaA_grasped'][idx]):
            if key: break
        key_idx_list.append(step_idx+1)
        key_states.append(data['obs'][idx][step_idx+1].astype(np.float32))
        for step_idx, k1 in enumerate(data['infos/is_cubeA_on_cubeB'][idx]):
            k2 = data['infos/is_cubaA_grasped'][idx][step_idx]
            if k1 and not k2: break
        # Right before such a state and so we do not use step_idx+1.
        key_idx_list.append(step_idx)
        key_states.append(data['obs'][idx][step_idx].astype(np.float32))

    # If PushChair (four key states):
    # key state I: right before demo_rotate -> true
    # key state II: right before demo_move -> true
    # key state III: when chair_close_to_target & chair_standing -> true
    # key state IV: end of the trajectory
    lengths = []
    # In PushChair, demo_* indicate the current state (not the next). 
    if task == 'PushChair-v1':
        for step_idx, key in enumerate(data['infos/demo_rotate'][idx]):
            if key: break
        lengths.append(step_idx)
        key_states.append(data['obs'][idx][step_idx].astype(np.float32))
        for step_idx, key in enumerate(data['infos/demo_move'][idx]):
            if key: break
        lengths.append(step_idx - np.sum(lengths))
        key_states.append(data['obs'][idx][step_idx].astype(np.float32))  
        for step_idx, key in enumerate(np.bitwise_and(
                data['infos/chair_close_to_target'][idx],
                data['infos/chair_standing'][idx])):
            if key: break
        lengths.append(step_idx + 1 - np.sum(lengths))
        key_states.append(data['obs'][idx][step_idx+1].astype(np.float32))
        lengths.append(len(data['infos/success'][idx]) - np.sum(lengths))

    # Always append the last state in the trajectory as the last key state.
    key_states.append(data['obs'][idx][-1].astype(np.float32))

    key_states = np.stack(key_states, 0).astype(np.float32)
    assert len(key_states) > 0, task
    return key_states, key_idx_list

def traj_obs(task_name):
    dataset = {}
    file_path = f"{DATA_PATH}/{task_name}/trajectory.state.pd_joint_delta_pos.h5"
    traj_all = h5py.File(file_path)
    ids = list(range(len(traj_all)))
    dataset['env_states'] = [np.array(
        traj_all[f"traj_{i}"]['env_states']) for i in ids]
    dataset['obs'] = [np.array(traj_all[f"traj_{i}"]["obs"]) for i in ids]
    dataset['actions'] = [np.array(traj_all[f"traj_{i}"]["actions"]) for i in ids]
    for k in traj_all['traj_0']['infos'].keys():
        dataset[f'infos/{k}'] = [np.array(
            traj_all[f"traj_{i}"]["infos"][k]) for i in ids]
        if k == 'info':
            for kk in traj_all['traj_0']['infos'][k].keys():
                dataset[f'infos/demo_{kk}'] = [np.array(
                    traj_all[f"traj_{i}"]["infos"][k][kk]) for i in ids]
    dataset['key_idx'] = [get_key_states(task_name, dataset, i)[-1] for i in ids]
    return dataset, len(traj_all)

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

def gaussian_weight(distance, sigma=0.5):
    return torch.exp(-distance**2 / (2 * sigma**2))

def efficient_batch_linear_interpolation(tensor1, tensor2):
    a, b = tensor1.shape  
    c = tensor2.shape[2]

    indices_floor = torch.floor(tensor1).long()
    indices_ceil = indices_floor + 1
    weights = tensor1 - indices_floor.float()

    indices_ceil = torch.clamp(indices_ceil, max=c-1)
    indices_floor = torch.clamp(indices_floor, max=c-1)

    gathered_floor = torch.gather(tensor2, 2, indices_floor.unsqueeze(-1)).squeeze(-1)
    gathered_ceil = torch.gather(tensor2, 2, indices_ceil.unsqueeze(-1)).squeeze(-1)

    interpolated = (1 - weights) * gathered_floor + weights * gathered_ceil
    return interpolated
        

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset,target=None):
        self.original_dataset = original_dataset.float()
        self.dim = original_dataset.shape[-1]
        assert target is None or isinstance(target, dict), "target must be dict or None"
        if isinstance(target, dict):
            target_list = []
            for i in range(len(target)):
                data_traj = target[f'traj_{i}']
                target_list.append(data_traj)
            self.target = target_list
        else:
            self.target = target
            
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data = self.original_dataset[idx]
        if self.target is not None:
            label = self.target[idx]
            return idx, data, label
        else:
            return idx, data, 0
        

def order_loss(tensor, beta):
    # Calculate the difference between adjacent elements
    diffs = tensor[1:] - tensor[:-1]
    negative_diffs = F.softplus(-diffs,beta = beta)
    loss = negative_diffs.mean(dim=1).sum()
    return loss


def efficient_temporal_loss(X):
    T, B = X.size()
    diff = X.unsqueeze(1) - X.unsqueeze(0)  # [T, T, B]
    diff_abs = diff.abs()
    triu_indices = torch.triu_indices(row=T, col=T, offset=1)
    selected_diff_abs = diff_abs[triu_indices[0], triu_indices[1], :]
    negative_diffs = F.softplus(-selected_diff_abs+2,beta = beta)
    loss = negative_diffs.sum() / (T * (T - 1) / 2 * B)*10
    return loss

def pad_and_stack(tensors, max_length):
    padded_tensors = []
    for t in tensors:
        last_val = t[-1]
        pad_tensor = last_val.repeat(max_length - len(t),1)
        padded_tensor = torch.cat((t, pad_tensor))
        padded_tensors.append(padded_tensor)
    
    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    
    task_list = ["PegInsertionSide-v0"]
    record_dict = defaultdict(list)
    len_mode = 1 # 0: truncation; 1: normalize to minimal length; 2: extend to maximal length
    for task_name in task_list:
        beta = 0.3
        model_mi_est = model_mi().cuda()
        model_mi_est = nn.DataParallel(model_mi_est)
        data, _ = traj_obs(task_name)
        data_list = data["obs"]
        data_list = [torch.from_numpy(arr) for arr in data_list]
        traj_len_list = [len(arr) for arr in data_list]
        min_len, max_len = min(traj_len_list), max(traj_len_list)
        if len_mode==0:
            batch = pad_sequence(data_list,batch_first=True)  
            batch = batch[:,:min_len,:].to(device) # [B, min_len, D]
        elif len_mode==1:
            data_list2 = []
            for tensor in data_list:
                tensor_b = F.interpolate(tensor.unsqueeze(dim=0).transpose(1,2), size=min_len, mode='linear').transpose(2,1).squeeze()
                data_list2.append(tensor_b)
            batch = torch.stack(data_list2).to(device)
        elif len_mode==2:
            batch = pad_and_stack(data_list,max_len).to(device)
            min_len = max_len
        else:
            raise NotImplementedError("This mode is not implemented yet.")

        num_concept = 5
        embed_size = 10
        dataset = IndexedDataset(batch)
        model = KFE_m(input_dim=dataset.dim+2*embed_size, output_dim=min_len, traj_len=min_len, text_len=num_concept, embed_size=embed_size).to(device)
        
        batch_size = 20
        epoch = 50
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)  
        optim = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-3)
        lr_scheduler = CosineAnnealingLRWarmup(optim, T_max=round(epoch*1000/batch_size), T_warmup=1)
        
        criterion = torch.nn.MSELoss(reduction='none') 
        time_range = torch.arange(min_len).to(device)
        concept_idx = torch.arange(num_concept).repeat_interleave(batch_size).to(device)
        
        for i in range(epoch):
            train_step = 0  
            for indices, samples, _ in dataloader:
                samples = samples.to(device)
                samples = samples.repeat(num_concept, 1, 1) 
                
                dist = model(samples, time_range, concept_idx)
                prob = F.softmax(dist,dim=-1)
                
                weighted_range = torch.sum(torch.mul(prob,time_range.unsqueeze(0)),dim=1)
                weighted_range = weighted_range.view(num_concept,-1) 
                loss_order = order_loss(weighted_range, beta)
                
                s_k_0 = batch_linear_interpolate2(samples, weighted_range.view(-1),shift=0)
                s_k_1 = batch_linear_interpolate2(samples, weighted_range.view(-1),shift=-2)
                mi = func_mi(s_k_0,s_k_1,model_mi_est)
                loss = loss_order - mi*4
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if (train_step%10==0):
                    print("Epoch: {}; Iteration: {} loss:{}".format((i),train_step, loss.item()))
                train_step += 1
                lr_scheduler.step()
                
        checkpoint_path = f"{MODEL_PATH}/{task_name}"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optim.state_dict(),}, checkpoint_path)
                                        