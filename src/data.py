import os
import numpy as np
import h5py
import pickle as pkl
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pickle

# Please specify the DATA_PATH (the base folder for storing data) in `path.py`.
from .path import DATA_PATH, PKL_PATH

def printname(name):
    print(name)
    
def select_n_unique_closest(input_list, n):
    if n > len(set(input_list)):
        raise ValueError("n must be less than or equal to the number of unique elements in the input list")
    if n <= 0:
        raise ValueError("n must be greater than 0")
    
    unique_sorted_list = sorted(set(input_list))

    step = (len(unique_sorted_list) - 1) / (n - 1)
    indices = [int(round(step * i)) for i in range(n)]

    selected_elements = [unique_sorted_list[i] for i in indices]
    
    return selected_elements

def select_n_unique_closest_by_value(input_list, n):
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if n == 1:
        return [sorted(set(input_list))[len(input_list) // 2]]
    if n==len(input_list):
        return input_list

    unique_sorted_list = sorted(set(input_list))
    if n > len(unique_sorted_list):
        raise ValueError("n must be less than or equal to the number of unique elements in the input list")

    min_val, max_val = unique_sorted_list[0], unique_sorted_list[-1]
    interval = (max_val - min_val) / (n - 1)

    selected_elements = [min_val]
    last_selected_val = min_val
    for _ in range(1, n-1):
        next_val = last_selected_val + interval
        closest_val = min(unique_sorted_list, key=lambda x: abs(x - next_val))
        selected_elements.append(closest_val)
        last_selected_val = closest_val
    selected_elements.append(max_val)

    if len(set(selected_elements)) < n:
        selected_elements = sorted(set(selected_elements))[:n]

    return selected_elements

class DemosFeature(Dataset):
    def __init__(self, file_path = f"{PKL_PATH}/demos_feature.pkl"):
        super().__init__()
        with open(file_path, 'rb') as f:
            self.data = pkl.load(f)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[f'traj_{index}']


# To obtain the padding function for sequences.
def get_padding_fn(data_names):
    assert 's' in data_names, 'Should at least include `s` in data_names.'

    def pad_collate(*args):
        assert len(args) == 1
        output = {k: [] for k in data_names}
        for b in args[0]:  # Batches
            for k in data_names:
                output[k].append(torch.from_numpy(b[k]))

        # Include the actual length of each sequence sampled from a trajectory.
        # If we set max_seq_length=min_seq_length, this is a constant across samples.
        output['lengths'] = torch.tensor([len(s) for s in output['s']])

        # Padding all the sequences.
        for k in data_names:
            output[k] = pad_sequence(output[k], batch_first=True, padding_value=0)

        return output

    return pad_collate

class Modified_MS2Demos(Dataset):
    def __init__(self, 
            data_split='train', 
            task='PickCube-v0', 
            obs_mode='state', 
            control_mode='pd_joint_delta_pos',
            length=-1,
            min_seq_length=None,
            max_seq_length=None,
            with_key_states=False,
            multiplier=20,  # Used for faster data loading.
            velocity = True,
            onlysim = "",
            IB = "",
            key_states = "abcdef",
            seed=None):  # seed for train/test spliting.
        self.task = task
        self.data_split = data_split
        self.seed = seed
        self.min_seq_length = min_seq_length  # For sampling trajectories.
        self.max_seq_length = max_seq_length  # For sampling trajectories.
        self.with_key_states = with_key_states  # Whether output key states.
        self.multiplier = multiplier
        self.control_mode = control_mode
        self.velocity = velocity

        # Usually set min and max traj length to be the same value.
        self.max_steps = -1  # Maximum timesteps across all trajectories.
        traj_path = os.path.join(DATA_PATH, f'{task}/trajectory.{obs_mode}.{control_mode}.h5')
        print('Traj path:', traj_path)
        self.data, self.ids = self.load_demo_dataset(traj_path, length)
        
        keystates_path = os.path.join(PKL_PATH, task, "concepts.pkl")
        with open(keystates_path, 'rb') as f:
            self.keyidx = pkl.load(f)
        
        self.keyidx = {key: select_n_unique_closest_by_value(value, len(key_states)-1) for key, value in self.keyidx.items()}

        # Cache key states for faster data loading.
        if self.with_key_states:
            self.idx_to_key_states = dict()

    def __len__(self):
        return len(self.data['env_states'])

    def __getitem__(self, index):
        # Offset by one since the last obs does not have a corresponding action.
        l = len(self.data['obs'][index]) - 1 
    
        # Sample starting and ending index given the min and max traj length.
        if self.min_seq_length is None and self.max_seq_length is None:
            s_idx, e_idx = 0, l
        else:
            min_length = 0 if self.min_seq_length is None else self.min_seq_length
            max_length = l if self.max_seq_length is None else self.max_seq_length
            assert min_length <= max_length
            if min_length == max_length:
                length = min_length
            else:
                length = np.random.randint(min_length, max_length, 1)[0]
            if length <= l:
                s_idx = np.random.randint(0, l - length + 1, 1)[0]
                e_idx = s_idx + length
            else:
                s_idx, e_idx = 0, l
        assert e_idx <= l, f'{e_idx}, {l}'

        # Call get_key_states() if you want to use the key states.
        # Here `s` is the state observation, `a` is the action, 
        # `env_states` not used during training (can be used to reconstruct env for debugging).
        # `t` is used for positional embedding as in Decision Transformer.
        data_dict = {
            's': self.data['obs'][index][s_idx:e_idx].astype(np.float32), 
            'a': self.data['actions'][index][s_idx:e_idx].astype(np.float32), 
            't': np.array([s_idx]).astype(np.float32),  
            # 'env_states': self.data['env_states'][index][s_idx:e_idx].astype(np.float32),
        }     
        if self.with_key_states:
            if f'key_states_{index}' not in self.idx_to_key_states:
                self.idx_to_key_states[f'key_states_{index}']  = self.get_key_states(index)
            data_dict['k'] = self.idx_to_key_states[f'key_states_{index}']
        return data_dict

    def info(self):  # Get observation and action shapes.
        return self.data['obs'][0].shape[-1], self.data['actions'][0].shape[-1]

    def load_demo_dataset(self, path, length):  
        dataset = {}
        traj_all = h5py.File(path)
        
        if length == -1:
            length = len(traj_all)
        np.random.seed(self.seed)  # Fix the random seed for train/test data split.

        # Since TurnFaucet uses 10 different faucet models, we shuffle the data
        # such that the resulting sampled data are evenly sampled across faucet models.
        if self.task == 'TurnFaucet-v0':
            ids = []
            for i in range(10):  # Hard-code the 10 data splits for permutation.
                t_ids = np.random.permutation(len(traj_all)//10)[:length//10]
                t_ids += i*len(traj_all)//10
                ids.append(t_ids)
            ids = np.concatenate(ids)
        # Since PushChair uses 5 different faucet models, we shuffle the data
        # such that the resulting sampled data are evenly sampled across chair models.
        elif self.task == 'PushChair-v1':
            ids = []
            for i in range(5):  # Hard-code the 5 data splits for permutation.
                t_ids = np.random.permutation(len(traj_all)//5)[:length//5]
                t_ids += i*len(traj_all)//5
                ids.append(t_ids)
            ids = np.concatenate(ids)
        else:           
            ids = np.random.permutation(len(traj_all))[:length]

        ids = ids.tolist() * self.multiplier  # Duplicate the data for faster loading.
        # ids = ids.tolist()

        # Note that the size of `env_states` and `obs` is that of the others + 1.
        # And most `infos` is for the next obs rather than the current obs.

        # `env_states` is used for reseting the env (might be helpful for eval)
        dataset['env_states'] = [np.array(
            traj_all[f"traj_{i}"]['env_states']) for i in ids]
        # `obs` is the observation of each step.
        dataset['obs'] = [np.array(traj_all[f"traj_{i}"]["obs"]) for i in ids]
        dataset['actions'] = [np.array(traj_all[f"traj_{i}"]["actions"]) for i in ids]

        # `rewards` is not currently used in CoTPC training.
        dataset['rewards'] = [np.array(traj_all[f"traj_{i}"]["rewards"]) for i in ids] 
        for k in traj_all['traj_0']['infos'].keys():
            dataset[f'infos/{k}'] = [np.array(
                traj_all[f"traj_{i}"]["infos"][k]) for i in ids]
            if k == 'info': # For PushChair.
                for kk in traj_all['traj_0']['infos'][k].keys():
                    dataset[f'infos/demo_{kk}'] = [np.array(
                        traj_all[f"traj_{i}"]["infos"][k][kk]) for i in ids]                

        self.max_steps = np.max([len(s) for s in dataset['env_states']])
        if not self.velocity:
            if self.control_mode == 'pd_joint_delta_pos': 
                dataset['obs'] = [np.concatenate([
                    np.array(traj_all[f"traj_{i}"]["obs"])[:, :9],
                    np.array(traj_all[f"traj_{i}"]["obs"])[:, 18:],
                ], -1) for i in ids]
            elif self.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
                slices = [slice(0,19), slice(38,41), slice(44,51), slice(57,82), slice(107,119)]
                dataset['obs'] = [np.concatenate([
                    np.array(traj_all[f"traj_{i}"]["obs"])[:, s] for s in slices
                ], -1) for i in ids]
        
        return dataset, ids

    def get_key_states(self, idx):
        new_idx = self.ids[idx]
        key_states_idx = self.keyidx[f'traj_{new_idx}']
        key_states = self.data['obs'][idx][key_states_idx].astype(np.float32)

        # Always append the last state in the trajectory as the last key state.
        key_states = np.concatenate([key_states, self.data['obs'][idx][-1:].astype(np.float32)], 0).astype(np.float32)
        assert len(key_states) > 0, self.task
        return key_states
 