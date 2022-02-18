import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
from tqdm import tqdm, trange

class Trader():
    def __init__(self, device='cuda'):
        self.device = device
    
    def trade(self, prices, trade_discount, trajectory, lo=0, hi=None, current=None):
        """
        Compute balance in portfolio along given trajectory, in batch.
        Args:
            `prices`: batch_size x num_days x num_assets
            `trade_discount`: num_assets
            `trajectory`: B x num_days x (num_assets - 1) with elements in. -1 for sell, 0 for maintain, 1 for buy
            `hi - lo` == num_days
        """
        batch_size = prices.shape[0]
        num_days = prices.shape[1]
        num_assets = prices.shape[2]
        
        if hi is None:
            hi = prices.shape[0]
        days = torch.arange(lo, hi)
        prices     = prices[:, days]
        trajectory = trajectory[:, days]
        
        if current is None:
            current = torch.from_numpy(np.asarray([1000] + [0] * (num_assets - 1)))
        
        # B x 3
        current = current.unsqueeze(0).tile((batch_size, 1))
        portfolio = torch.zeros((batch_size, num_days + 1, num_assets))
        
        # B x num_assets x num_assets
        eye = torch.eye(num_assets, device=self.device).unsqueeze(0).tile((batch_size, 1, 1))
        for i in trange(lo, hi):
            left_matrix = torch.concat([
                - (trade_discount * prices[:, i]).unsqueeze(-2),    # B x 1 x A
                eye                                                 # B x A x A
            ])
            portfolio[:, i] = current + torch.bmm(left_matrix, prices[:, i].unsqueeze(-1))
            current = portfolio[:, i]
            
            tqdm.write(f"Currently the max trajectory produces { (current * prices[:, i]).sum(-1).max().item() } in portfolio.")
            
        return portfolio