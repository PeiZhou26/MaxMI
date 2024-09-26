import argparse, os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idx", default="0", type=str, help="which gpu to be used")
    return parser.parse_args()

import glob
import torch
import math
import time
import logging
import pickle
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata

from torch import nn
from torch import optim
from torch.optim import Adam
import numpy as np
import torch.autograd as autograd

from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO
from perceiver.query import Query_Gen
from perceiver.query_new import Query_Gen_transformer, Query_Gen_transformer_PE
from util.epoch_timer import epoch_time, scale_data, mean_variance_loss
from util.look_table import lookup_value_2d, lookup_value_close, lookup_value_average, lookup_value_bilinear, lookup_value_grid

from perceiver.encoder import PerceiverEncoder
from perceiver.decoder import PerceiverDecoder
from perceiver.perceiver import PerceiverIO
from perceiver.perceiver_lap import PerceiverIO_lap
import os, sys
import matplotlib.pyplot as plt
from scipy.special import digamma
from scipy.spatial import KDTree
import scipy.special

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SampleDataset(Dataset):
    def __init__(self, sample_x, sample_y):
        self.dataX, self.dataY = sample_x, sample_y

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]

def append_to_file(filename, text_to_append):
    with open(filename, "a") as file:  
        file.write(f"{text_to_append}\n")  

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def infer(model, batch):
    # model.eval()
    model.train() 
    if isinstance(batch, np.ndarray):
        batch = torch.tensor(batch, dtype=torch.float32).cuda()
    mi_lb = model(batch)
    return mi_lb

def compute_smi_mean(sample_x, sample_y, model, proj_num, num_per_iter):
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    results = []
    seq_len = sample_x.shape[0]
    # num_per_iter = 4
    for i in range(proj_num//num_per_iter):
        batch = np.zeros((num_per_iter, seq_len, 2))
        for j in range(num_per_iter):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            x_proj = rankdata(x_proj)/seq_len
            y_proj = rankdata(y_proj)/seq_len
            xy = np.column_stack((x_proj, y_proj))
            # xy = scale_data(xy)
            batch[j, :, :] = xy
        infer1 = infer(model, batch).cpu().numpy()
        print(f"iter: {i}")
        mean_infer1 = np.mean(infer1)
        results.append(mean_infer1)
    return np.mean(np.array(results))

def compute_smi_mean_50(sample_x, sample_y, model, proj_num, num_per_iter):
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    seq_len = sample_x.shape[0]
    
    if isinstance(sample_x, np.ndarray):
        batch = np.zeros((dx, seq_len, 2))
    elif isinstance(sample_x, torch.Tensor):
        batch = torch.zeros((dx, seq_len, 2))
    else:
        raise TypeError("Unsupported data type")
    for j in range(dx):
        x_proj = sample_x[:,j]
        y_proj = sample_y[:,j]
        if (x_proj.max()==x_proj.min()) or (y_proj.max()==y_proj.min()):
            if isinstance(x_proj, np.ndarray):
                batch[j, :, :] = np.zeros((seq_len,2))
            elif isinstance(x_proj, torch.Tensor):
                batch[j, :, :] = torch.zeros((seq_len,2))
            else:
                raise TypeError("Unsupported data type")
        else:
            x_proj = (x_proj-x_proj.min())/(x_proj.max()-x_proj.min())
            y_proj = (y_proj-y_proj.min())/(y_proj.max()-y_proj.min())
            if isinstance(x_proj, np.ndarray):
                xy = np.column_stack((x_proj, y_proj))
            elif isinstance(x_proj, torch.Tensor):
                xy = torch.stack((x_proj, y_proj), dim=-1)
            else:
                raise TypeError("Unsupported data type")
            batch[j, :, :] = xy

    batch = batch.cuda()
    mi = infer(model, batch)
    return torch.mean(mi)

def compute_smi_mean_uniform(sample_x, sample_y, model, proj_num, num_per_iter):
    dx = sample_x.shape[1]
    dy = sample_y.shape[1]
    proj_num = 1000
    model = nn.Linear(dx, proj_num)
    dataset = SampleDataset(sample_x, sample_y)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=True, drop_last=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 200
    for epoch in range(epochs):
        for i, (inputs1, inputs2) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs1 = model(inputs1)
            outputs2 = model(inputs2)
            loss = mean_variance_loss(outputs1) + mean_variance_loss(outputs2) 
            loss.backward()
            optimizer.step()
        if epoch%10==0:
            print(loss.item())
        
    sample_x = model(sample_x).detach().permute(1,0)
    tmp = 2*(sample_x - sample_x.min(dim=1, keepdim=True)[0]) / (sample_x.max(dim=1, keepdim=True)[0] - sample_x.min(dim=1, keepdim=True)[0])-1 
    sample_y = model(sample_y).detach().permute(1,0)

    results = []
    seq_len = sample_x.shape[0]
    # num_per_iter = 4
    for i in range(proj_num//num_per_iter):
        batch = np.zeros((num_per_iter, seq_len, 2))
        for j in range(num_per_iter):
            theta = np.random.randn(dx)
            phi = np.random.randn(dy)
            x_proj = np.dot(sample_x, theta)
            y_proj = np.dot(sample_y, phi)
            xy = np.column_stack((x_proj, y_proj))
            xy = scale_data(xy)
            batch[j, :, :] = xy
        infer1 = infer(model, batch).cpu().numpy()
        print(f"iter: {i}")
        mean_infer1 = np.mean(infer1)
        results.append(mean_infer1)
    return np.mean(np.array(results))

def model_mi():
    latent_dim = 256
    latent_num = 256
    input_dim = 2
    decoder_query_dim = 1000

    encoder = PerceiverEncoder(
        input_dim=input_dim,
        latent_num=latent_num,
        latent_dim=latent_dim,
        cross_attn_heads=8,
        self_attn_heads=8,
        num_self_attn_per_block=8,
        num_self_attn_blocks=1
    )

    decoder = PerceiverDecoder(
        q_dim=decoder_query_dim,
        latent_dim=latent_dim,
    )

    query_gen = Query_Gen_transformer(
        input_dim = input_dim,
        dim = decoder_query_dim
    )

    model = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).cuda()
    model.load_state_dict(torch.load('your_path_of_mi_estimator', map_location="cpu"))
    return model

def func_mi(x,y,model): 
    proj_num = 40000
    result = compute_smi_mean_50(x, y, model, proj_num=proj_num,num_per_iter=100)
    return result

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    
    latent_dim = 256
    latent_num = 256
    input_dim = 2
    decoder_query_dim = 1000

    encoder = PerceiverEncoder(
        input_dim=input_dim,
        latent_num=latent_num,
        latent_dim=latent_dim,
        cross_attn_heads=8,
        self_attn_heads=8,
        num_self_attn_per_block=8,
        num_self_attn_blocks=1
    )

    decoder = PerceiverDecoder(
        q_dim=decoder_query_dim,
        latent_dim=latent_dim,
    )

    query_gen = Query_Gen_transformer(
        input_dim = input_dim,
        dim = decoder_query_dim
    )

    model = PerceiverIO(encoder=encoder, decoder=decoder, query_gen = query_gen, decoder_query_dim = decoder_query_dim).cuda()
    model.load_state_dict(torch.load('your_path_of_mi_estimator', map_location="cpu"))
    print(f'The model has {count_parameters(model):,} trainable parameters')