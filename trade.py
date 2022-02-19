import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn

from models.Trader import Seq2SeqPolicy
from utils import *

data_path = "data/data.csv"

def optimize(pf, price_seq, tradability, net_fn: Seq2SeqPolicy, costs, gamma=0.99):
    """
    Args:
        `portfolio`: B x num_assets
        `prices_seq`: B x seq_len x num_assets
        `tradability`: B x seq_len x num_assets
        `costs`: num_assets
    """
    batch_size = price_seq.shape[0]
    seq_len = price_seq.shape[1]
    num_assets = price_seq.shape[2]
    
    # B x s x 2, B x s x 3
    sell, buy = net_fn(pf, price_seq, tradability)
    
    profit = torch.zeros([batch_size], device=pf.device)
    
    price_seq = torch.concat([
        price_seq,
        price_seq[:, -1:]
    ], dim=1)
    
    for date in range(seq_len):
        new_pf = pf
        
        # Sell
        assets_trade = pf[:, 1:] * sell[:, date]
        new_cash = pf[:, 0] + (assets_trade * tradability[:, date] * (1 - costs) * price_seq[:, date]).sum(-1)
        
        # Buy
        cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, date]
        new_assets = cash_trade[:, :2] * tradability[:, date] * (1 - costs) / price_seq[:, date]
        
        new_pf = torch.concat([
            cash_trade[:, -1:],
            new_assets
        ], dim=-1)
        
        value     = pf[:, 0] + (pf[:, 1:] * price_seq[:, date]).sum(-1)
        new_value = new_pf[:, 0] + (new_pf[:, 1:] * price_seq[:, date + 1]).sum(-1)
        
        profit = profit + (new_value - value) * gamma
        gamma *= gamma
        
    return profit

def main(args, writer=None):
    prices, tradability = get_data(args)
    
    num_assets = len(args.assets)
    num_days = prices.shape[0]
    
    net = Seq2SeqPolicy(num_assets, args.seq_len, consider_tradability=True, device=args.device)
    
    init_pf = torch.zeros([args.batch_size, num_assets + 1], device=args.device)
    init_pf[..., 0] = args.initial_cash
    
    for epoch in range(args.epoch):
        costs = torch.from_numpy(np.asarray([args.cost_trans[asset] for asset in args.assets])).to(args.device).unsqueeze(0)
        
        for step in trange(num_days - args.seq_len):
            optimizer = torch.optim.SGD(params=net.parameters(), lr=1e-3 * (0.999**epoch))
            
            seq_start_idx = torch.randint(0, num_days - args.seq_len, [args.batch_size], device=args.device)
            price_seq = torch.zeros([0, args.seq_len, num_assets], device=args.device)
            tdblt_seq = torch.zeros([0, args.seq_len, num_assets], device=args.device)
            
            for batch in range(args.batch_size):
                price_seq = torch.concat([price_seq, prices[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len].clone().unsqueeze(0)], dim=0)
                tdblt_seq = torch.concat([tdblt_seq, tradability[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len].clone().unsqueeze(0)], dim=0)
            
            price_seq = prices[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
            tdblt_seq = tradability[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
            
            profit = optimize(init_pf, price_seq, tdblt_seq, net, costs=costs, gamma=args.profit_discount)
            tqdm.write(f"Step #{ step }: Ave. profit { profit.mean(dim=-1).item() }")
            
            optimizer.zero_grad()
            loss = - profit.sum(-1)
            epsilon = torch.rand(1)
            if epsilon < 0.4 * np.exp(-0.1 * epoch):
                loss = -loss * torch.rand(1).cuda()
            loss.backward()
            optimizer.step()
    
def parse_arguments(agile=False):
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    parser.add_argument("--profit_discount", default=0.95, type=float, help="Discount of profit computation")
    
    # Data
    parser.add_argument("--seq_len", default=32, type=str, help="Length of sequence")
    parser.add_argument("--data_path", default="data/data.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--epoch", default=256, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--steps", default=4096, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", type=str, help="Device of computation")
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    main(args)
