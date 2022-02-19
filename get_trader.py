import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn

from models.Policy import Seq2SeqPolicy
from torch.utils.tensorboard import SummaryWriter
from utils.data import *

# TODO: Fix this...
def sample_tradability(batch_size, seq_len, num_assets=2, indices=[-1], rate=0.2, device='cuda'):
    tradability = torch.ones([batch_size, num_assets], device=device).float()
    for idx in indices:
        untradable_idx = torch.randint(0, len, [len * 0.3], device=device)
        # untradable_idx = torch.randint(0, len, [len * 0.3 * torch.rand((1))], device=device)
        tradability[idx, untradable_idx] = 0.
        
    return tradability
    
def sample_init_pf(batch_size, num_assets=2, ranges=[[0., 200000.], [0., 100.], [0., 2000.]], device='cuda'):
    ranges =  torch.Tensor(ranges).to(device)
    ran_len = ranges[:, 1] - ranges[:, 0]
    init_pf = torch.randint([batch_size, num_assets + 1], device=device)
    init_pf = init_pf * ran_len - ranges[:, 0]
    
    return init_pf.detach()

def train(step, pf, price_seq, tradability, net_fn: Seq2SeqPolicy, costs, gamma=0.99, plot_trade=False, writer=None):
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
    
    init_value = pf[:, 0] + (pf[:, 1:] * price_seq[:, 0]).sum(-1).detach()
    
    # B x s x 2, B x s x 3
    sell, buy = net_fn(pf, price_seq, tradability)
    
    # reg_vec = torch.concat([sell, buy], dim=-1).reshape((-1, num_assets * 2 + 1))
    # reg = (reg_vec * (1 - reg_vec)).sum(-1)
    
    if plot_trade:
        plot = {
            "Cash": pf[0, 0].detach().cpu(),
            "BTC":  pf[0, 1].detach().cpu(),
            "Gold": pf[0, 2].detach().cpu(),
            "Price_BTC": price_seq[0, 0, 0].detach().cpu(),
            "Price_Gold": price_seq[0, 0, 1].detach().cpu(),
            "Value": init_value[0].detach().cpu(),
            "Gamma": gamma
        }
        writer.add_scalars(f"Trade { step }", plot, 0)
    
    profit = torch.zeros([batch_size], device=pf.device)
    new_pf = pf.clone()
    for date in range(seq_len - 1):
        # Sell
        assets_trade = new_pf[:, 1:] * sell[:, date]
        new_cash = new_pf[:, 0] + (assets_trade * tradability[:, date] * (1 - costs) * price_seq[:, date]).sum(-1)
        
        # Buy
        cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, date]
        new_assets = cash_trade[:, 1:] * tradability[:, date] * (1 - costs) / price_seq[:, date] + new_pf[:, 1:] * (1 - sell[:, date])
        
        new_pf = torch.concat([
            cash_trade[:, :1],
            new_assets
        ], dim=-1)
        
        new_value = new_pf[:, 0] + (new_pf[:, 1:] * price_seq[:, date + 1]).sum(-1)
        
        # profit = profit + (new_value - value) * gamma
        profit = profit + (new_value - init_value) * gamma
      
        if plot_trade:
            plot = {
                "Cash": cash_trade[0, 0].detach().cpu(),
                "BTC":  new_pf[0, 1].detach().cpu(),
                "Gold": new_pf[0, 2].detach().cpu(),
                "Price_BTC": price_seq[0, date, 0].detach().cpu(),
                "Price_Gold": price_seq[0, date, 1].detach().cpu(),
                "Value": new_value[0].detach().cpu(),
                "Gamma": gamma
            }
            writer.add_scalars(f"Trade { step }", plot, date + 1)
        
        gamma *= gamma
        
    return profit.sum(-1), new_value.sum(-1), reg

def get_trader(args, writer=None): 
    net = Seq2SeqPolicy(num_assets, args.seq_len, consider_tradability=args.consider_tradability, seq_mode=args.seq_encode_mode, device=args.device)
    sd_filename = os.path.join("data", "trader", f"{args.seq_len}_{ args.seq_encode_mode }_{ 'cons' if args.consider_tradability else 'ignr' }.pt")
    
    if args.force_retrain or not os.path.isfile(sd_filename):
        seq = torch.from_numpy(pd.read_csv(args.seq_file_path).to_numpy()).to(args.device)
        prices, tradability = get_data(args)
        
        num_assets = len(args.assets)
        num_days = prices.shape[0]
        
        net.train()
        
        for epoch in range(args.epoch):
            prof_mean = 0
            val_mean  = 0
            costs = torch.from_numpy(np.asarray([args.cost_trans[asset] for asset in args.assets])).to(args.device).unsqueeze(0)
            
            optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3*(0.999**epoch))
            for step in trange(args.steps):
                
                # B x seq_len x num_assets
                prices = seq_slide_select(seq, args.batch_size, args.seq_len, len(args.assets))
                # B x (num_assets + 1)
                init_pf = sample_init_pf(args.batch_size, num_assets, ranges=[[0., 200000.], [0., 100.], [0., 2000.]], device='cuda')
                tradability = sample_tradability(args.batch_size, args.seq_len, num_assets, indices=[-1], rate=0.2, device='cuda')
                # Training with real data - not used
                # init_pf = torch.zeros([args.batch_size, num_assets + 1], device=args.device).float()
                # init_pf[..., 0] = args.initial_cash * torch.rand([1], device=args.device).float()

                seq_start_idx = torch.randint(0, num_days - args.seq_len, [args.batch_size], device=args.device)
                price_seq = torch.zeros([0, args.seq_len, num_assets], device=args.device)
                tdblt_seq = torch.zeros([0, args.seq_len, num_assets], device=args.device)
                
                for batch in range(args.batch_size):
                    price_seq = torch.concat([price_seq, prices[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len].clone().unsqueeze(0)], dim=0)
                    tdblt_seq = torch.concat([tdblt_seq, tradability[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len].clone().unsqueeze(0)], dim=0)
                
                # price_seq = prices[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
                # tdblt_seq = tradability[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
                
                profit, value, reg = train(step, init_pf, price_seq, tdblt_seq, net, costs=costs, gamma=args.profit_discount, plot_trade=False, writer=writer)
                
                optimizer.zero_grad()
                loss = - profit
                # loss = - profit + reg.sum(-1)
                epsilon = torch.rand(1)
                if epsilon < 0.4 * np.exp(-0.5 * epoch):
                    loss = - loss * torch.rand(1).cuda()
                loss.backward()
                optimizer.step()
                
                prof_mean += profit.detach()
                val_mean  += value.detach()
        
            tqdm.write(f"Epoch #{ epoch }: Ave. profit { ( prof_mean / args.steps ).item() } Ave. value { ( val_mean / args.steps ).item() }")
            
        torch.save(net.state_dict, sd_filename)
    else:
        net.load_state_dict(torch.load(sd_filename))
        
    return net
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    parser.add_argument("--profit_discount", default=0.95, type=float, help="Discount of profit computation")
    
    # Trader
    parser.add_argument("--force_retrain", default=False, type=bool, help="Whether or not forcely retrain the agent")
    parser.add_argument("--consider_tradability", default=True, type=bool, help="Whether or not feed tradability into the network")
    parser.add_argument("--seq_encode_mode", default="gru", type=str,
                        help="Mode of encoding price sequence: ['deri-1', 'delta', 'gru']")
    
    # Data
    parser.add_argument("--seq_len", default=16, type=int, help="Length of sequence")
    parser.add_argument("--data_path", default="data/data.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--epoch", default=40, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--steps", default=512, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", type=str, help="Device of computation")
    
    # Debug
    parser.add_argument("--seq_file_path", default="data/gen_seq.csv", type=str, help="Path of generated sequence file")
    parser.add_argument("--summary_log_dir", default="runs/", type=str, help="Log directory for Tensorboard Summary Writer")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    writer = SummaryWriter(args.summary_log_dir)
    get_trader(args, writer)
    