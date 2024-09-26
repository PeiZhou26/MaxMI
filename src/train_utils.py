"""
Code for the cosine decay learning rate with linear warmup.
"""

import math
import functools
import torch

def mse_loss_with_weights(preds, targets, weights=None):
    losses = torch.mean((preds - targets) ** 2, -1)
    if weights is None:
        return torch.mean(losses)
    else:
        assert losses.shape == weights.shape, losses.shape
        return torch.mean(losses * weights)

def get_loss(preds, targets, lengths):    
    # If we have sequences of varied lengths, use masks so we do not compute loss 
    # over padded values. If we set max_seq_length=min_seq_length, then it should 
    # not matter since all sequences have the same length.
    B = preds.shape[0]
    max_len = torch.max(lengths)  # Max length of the current mini-batch.
    lengths = lengths[:, None]  # B x 1
    temp = torch.arange(0, max_len)[None].expand(B, -1).cuda()  # B x max_len
    masks = (temp < lengths.expand(B, max_len)).float() # B x max_len

    loss = mse_loss_with_weights(
        preds.reshape(-1, preds.size(-1)), 
        targets.reshape(-1, targets.size(-1)),
        masks.reshape(-1))
    return loss

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.1
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = max(0.1, 0.5 * (1 + math.cos(math.pi * multiplier)))
    return multiplier


def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
