import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
from tqdm import tqdm, trange


def get_seq_gen(device='cuda'):
    return torch.from_numpy(pd.read_csv("data/seq_gen.csv").to_numpy().transpose((-1, -2)))[1].to(device)

# def price_tradability_seq_from_seq_gen(num_assets, seq_len, tradability_assets=None, device='cuda'):
#     num_assets = 
    
#     seqs = get_seq_gen(device=device).unsqueeze(0).unsqueeze(-1).tile((batch_size, -1, num_assets))
#     return seq_slide_select(seqs, seq_len, require_tradability=True, tradability_assets=tradability_assets)
    
# def get_grader_seq_data(filepath, seq_len, device='cuda'):
#     # seq_len
#     sequence = torch.from_numpy(pd.read_csv(filepath).to_numpy()).to(device)

def seq_slide_select(sequences, seq_len, require_tradability=False, tradability_assets=None):
    """
    Args:
        `sequences`: B x seq_len x num_assets
    """
    batch_size = sequences.shape[0]
    len        = sequences.shape[1]
    num_assets = sequences.shape[2]
    
    start_idx = torch.randint(0, len - seq_len, [batch_size, num_assets], device=sequences.device)
    seqs = torch.zeros([0, seq_len, num_assets], device=sequences.device)
    
    for batch in range(batch_size):
        seq = torch.zeros([seq_len, 0], device=sequences.device)
        for asset in range(num_assets):
            seq = torch.concat([
                seq,
                sequences[batch, start_idx[batch, asset]:start_idx[batch, asset]+seq_len, asset].unsqueeze(-1),
            ], dim=-1)
        seqs = torch.concat([
            seqs,
            seq.unsqueeze(0)
        ], dim=0)
        
    if require_tradability:
        tradability = torch.ones_like(seqs)
        untradable_idx = torch.randint(0, len, [batch_size, int(len * 0.3)], device=sequences.device)
        for batch in range(batch_size):
            for idx in tradability_assets:
                # untradable_idx = torch.randint(0, len, [len * 0.3 * torch.rand((1))], device=device)
                tradability[batch, untradable_idx[batch], idx] = 0.
        
        return seqs, tradability
    else:
        return seqs

def get_data(data_path, device):
    if not os.path.isfile(data_path):
        raise NotImplementedError()
    print(f"Reading data from file { data_path }")
    
    data = pd.read_csv(data_path)
    
    # num_days x num_assets with prices in USD
    df = pd.DataFrame(data=data, columns=['btc', 'gold_inter'])
    prices = torch.from_numpy(df.to_numpy()).float().to(device)
    # num_days x num_assets with {0., 1.}
    tradability = torch.from_numpy(pd.DataFrame(data=data, columns=['btc_tradable', 'gold_tradable']).to_numpy()).float().to(device)
    
    print(f"========== Data Loaded ==========")
    print(df.describe())
    print(f"Totally { data.shape[0] } days of trade, with { torch.from_numpy(data['gold_tradable'].to_numpy() == False).int().sum().item() } unavailable for gold.")
    print(f"========== Data Loaded ==========")
    return prices, tradability

"""
Abandoned
"""
class TradeComputation():
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
