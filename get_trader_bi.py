import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch
import torch.nn as nn
from models.Forecaster import Forecaster

from models.Policy import Seq2SeqPolicy, BiSeq2SeqPolicy
from torch.utils.tensorboard import SummaryWriter
from utils.data import *

# TODO: Fix this...
def sample_tradability(seq_len, num_assets=2, indices=[-1], rate=0.2, device='cuda'):
    tradability = torch.ones([seq_len, num_assets], device=device).float()
    for idx in indices:
        untradable_idx = torch.randint(0, seq_len, [int(seq_len * 0.3)], device=device)
        # untradable_idx = torch.randint(0, len, [len * 0.3 * torch.rand((1))], device=device)
        tradability[untradable_idx, idx] = 0.
        
    return tradability
    
def sample_init_pf(batch_size, num_assets=2, ranges=[[0., 200000.], [0., 100.], [0., 2000.]], device='cuda'):
    ranges =  torch.Tensor(ranges).to(device)
    ran_len = ranges[:, 1] - ranges[:, 0]
    init_pf = torch.rand([batch_size, num_assets + 1], device=device)
    init_pf = init_pf * ran_len - ranges[:, 0]
    
    return init_pf.detach()

def run_network(step, pf, prices_fc, prices_gt, tradability, net_fn: Seq2SeqPolicy, costs, gamma=0.99, plot_trade=False, ret_pf=False, writer=None):
    """
    Args:
        `portfolio`: B x num_assets
        `prices_fc`: forecasted   B x seq_len x num_assets
        `prices_gt`: ground truth B x seq_len x num_assets
        `tradability`: B x seq_len x num_assets
        `costs`: num_assets
    """
    batch_size = prices_gt.shape[0]
    seq_len = int(prices_gt.shape[1] / 2)
    num_assets = prices_gt.shape[2]
    
    init_value = pf[:, 0] + (pf[:, 1:] * prices_gt[:, 0]).sum(-1).detach()
    
    # B x s x 2, B x s x 3
    sell, buy = net_fn(pf, prices_fc, tradability)
    
    # reg_vec = torch.concat([sell, buy], dim=-1).reshape((-1, num_assets * 2 + 1))
    # reg = (reg_vec * (1 - reg_vec)).sum(-1)
    
    profit = torch.zeros([batch_size], device=pf.device)
    new_pf = pf.clone()
    for date in range(seq_len - 1):
        # Sell
        assets_trade = new_pf[:, 1:] * sell[:, date]
        new_cash = new_pf[:, 0] + (assets_trade * tradability[:, date + seq_len] * (1 - costs) * prices_gt[:, date + seq_len]).sum(-1)
        
        # Buy
        cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, date]
        new_assets = cash_trade[:, 1:] * tradability[:, date + seq_len] * (1 - costs) / prices_gt[:, date + seq_len] + new_pf[:, 1:] * (1 - sell[:, date])
        
        new_pf = torch.concat([
            cash_trade[:, :1],
            new_assets
        ], dim=-1)
        
        # profit = profit + (new_value - init_value) * gamma
        # gamma *= gamma
        new_value = new_pf[:, 0] + (new_pf[:, 1:] * prices_gt[:, date + seq_len + 1]).sum(-1)
        profit = profit + (new_value - init_value) * gamma
        
        if plot_trade:
            plot = {
                "Cash": cash_trade[0, 0].detach().cpu(),
                "BTC":  new_pf[0, 1].detach().cpu(),
                "Gold": new_pf[0, 2].detach().cpu(),
                "Price_BTC": prices_gt[0, date, 0].detach().cpu(),
                "Price_Gold": prices_gt[0, date, 1].detach().cpu(),
                "Price_BTC_FC": prices_fc[0, date, 0].detach().cpu(),
                "Price_Gold_FC": prices_fc[0, date, 1].detach().cpu(),
                "Value": new_value[0].detach().cpu(),
                "Gamma": gamma
            }
            writer.add_scalars(f"Trade", plot, step + date)
        
        
    if ret_pf:
        return profit, new_value - init_value, new_value, new_pf
    else:
        return profit, new_value - init_value


def e2e_run(args, net, plot_trade=True, writer=None):
    prices, tradability = get_data(args.real_data_path, args.device)
    
    batch_size = 1
    num_assets = len(args.assets)
    num_days = prices.shape[0]
    init_value = 1000.
    
    costs = torch.from_numpy(np.asarray([args.cost_trans[asset] for asset in args.assets])).to(args.device).unsqueeze(0)
    
    # Post-padding for tradability, not needed for prices.
    tradability = torch.concat([tradability, tradability[-args.seq_len:]], dim=0).unsqueeze(0)
    # prices = torch.concat([prices, prices[-args.seq_len:]], dim=0).unsqueeze(0)
    
    # prices = prices + torch.normal(0, 1, size=prices.shape, device=prices.device) * prices.mean() * 0.05
    
    # Forecaster for prices
    forecaster = Forecaster(prices)
    
    net.eval()
    
    pf = torch.tensor([1000., 0., 0.]).to(args.device).unsqueeze(0)
    
    if args.seq_stripe:
        prof = 0
        for step in tqdm(np.arange(args.initial_waiting, num_days, args.seq_len - 1)):
            prices_fc, prices_gt = forecaster.forecast(step, args.seq_len, mode='stat', padding=True)
            prices_fc = torch.concat([prices[step - args.seq_len:step], prices_fc], dim=0).unsqueeze(0)
            prices_gt = torch.concat([prices[step - args.seq_len:step], prices_gt], dim=0).unsqueeze(0)
            
            # _, new_value, pf = run_network(step, pf,
            reward, profit, new_value, pf = run_network(step, pf,
                                 prices_fc, prices_gt, tradability[:, step:step+args.seq_len],
                                #  prices[:, step:step+args.seq_len], tradability[:, step:step+args.seq_len],
                                 net, costs=costs, gamma=args.profit_discount,
                                 plot_trade=True, ret_pf=True, writer=writer)
            prof = prof + profit.sum(-1)
            tqdm.write(f"Step { step }: value = { new_value.item() }, pf = { pf.detach().cpu().numpy() }, pr = { prices[step].detach().cpu().numpy() }")
        # new_value = new_value.sum(-1)
    else:
        for step in trange(args.initial_waiting, num_days):
            prices_fc, prices_gt = forecaster.forecast(step, args.seq_len, mode='ema', padding=True)
            prices_fc = torch.concat([prices[step - args.seq_len:step], prices_fc], dim=0).unsqueeze(0)
            prices_gt = torch.concat([prices[step - args.seq_len:step], prices_gt], dim=0).unsqueeze(0)
            
            # prices = prices + torch.normal(0, 1, size=prices.shape, device=prices.device) * prices.mean() * 0.05
            sell, buy = net(pf, prices_fc, tradability[:, step:step+args.seq_len])
            # Sell
            assets_trade = pf[:, 1:] * sell[:, 0]
            new_cash = pf[:, 0] + (assets_trade * tradability[:, step] * (1 - costs) * prices_gt[:, args.seq_len]).sum(-1)
            
            # Buy
            cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, 0]
            new_assets = cash_trade[:, 1:] * tradability[:, step] * (1 - costs) / prices_gt[:, args.seq_len] + pf[:, 1:] * (1 - sell[:, 0])
            
            pf = torch.concat([
                cash_trade[:, :1],
                new_assets
            ], dim=-1)
            
            new_value = (pf[:, 0] + (pf[:, 1:] * prices_gt[:, args.seq_len + 1]).sum(-1))
            
            if plot_trade:
                plot = {
                    "Cash": cash_trade[0, 0].detach().cpu(),
                    "BTC":  pf[0, 1].detach().cpu(),
                    "Gold": pf[0, 2].detach().cpu(),
                    "Price_BTC": prices[step, 0].detach().cpu(),
                    "Price_Gold": prices[step, 1].detach().cpu(),
                    "Value": new_value[0].detach().cpu()
                }
                writer.add_scalars(f"Trade", plot, step + 1)
            tqdm.write(f"Step { step }: value = { new_value[0].item() }, pf = { pf.detach().cpu().numpy() }")
        new_value = new_value[0]
    
    # new_value = new_value.detach().cpu().item()
    print(f"""
========== Evaluation ==========
Value: { new_value.item() } with profit { new_value.item() - init_value }
================================
          """)
        
def get(args, writer=None): 
    net = BiSeq2SeqPolicy(len(args.assets), args.seq_len, consider_tradability=args.consider_tradability, seq_mode=args.seq_encode_mode, output_daily=args.output_daily, device=args.device)
    sd_filename = os.path.join("data", "trader", f"bi_{args.seq_len}_{ args.seq_encode_mode }_{ 'cons' if args.consider_tradability else 'ignr' }_{ args.epoch }.pt")
    
    if args.force_retrain or not os.path.isfile(sd_filename):
        prices, tradability = get_data(args.data_path, args.device, sample_tradability=True, untradability_assets=[-1])
        prices = prices / 10 + 200 + 500 * torch.rand((1)).to(args.device)
        prices = prices + torch.normal(0, 1, size=prices.shape, device=prices.device) * prices.mean() * 0.05
        
        num_days = prices.shape[0]
        num_assets = len(args.assets)
        
        tradability = sample_tradability(num_days, num_assets, indices=[-1], rate=0.2)
        
        net.train()
        
        costs = torch.from_numpy(np.asarray([args.cost_trans[asset] for asset in args.assets])).to(args.device).unsqueeze(0)
        for epoch in range(args.epoch):
            prof_mean = 0
            rewd_mean  = 0
            
            optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3*(0.999**epoch))
            for step in trange(args.steps):
                # B x seq_len x num_assets
                # prices, tradability = seq_slide_select(seq, args.seq_len, require_tradability=True, tradability_assets=[-1])
                # B x (num_assets + 1)
                init_pf = sample_init_pf(args.batch_size, num_assets, ranges=[[0., 20000.], [0., 10.], [0., 200.]], device='cuda')
                # Training with real data - not used
                # init_pf = torch.rand([args.batch_size, num_assets + 1], device=args.device).float()
                # init_pf[..., 0] = args.initial_cash * torch.rand([1], device=args.device).float() * 100

                seq_start_idx = torch.randint(0, num_days - args.seq_len * 2, [args.batch_size], device=args.device)
                price_seq = torch.zeros([0, args.seq_len * 2, num_assets], device=args.device)
                tdblt_seq = torch.zeros([0, args.seq_len * 2, num_assets], device=args.device)
                
                for batch in range(args.batch_size):
                    price_seq = torch.concat([price_seq, prices[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len * 2].clone().unsqueeze(0)], dim=0)
                    tdblt_seq = torch.concat([tdblt_seq, tradability[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len * 2].clone().unsqueeze(0)], dim=0)
                
                # price_seq = prices[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
                # tdblt_seq = tradability[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
                
                reward, profit = run_network(step, init_pf, price_seq, price_seq, tdblt_seq, net, costs=costs, gamma=args.profit_discount, plot_trade=False, writer=writer)
                
                optimizer.zero_grad()
                loss = - ((reward + profit).sum(-1) + (profit.mean() / profit.std()))
                          
                epsilon = torch.rand(1)
                if epsilon < 0.4 * np.exp(-0.5 * epoch):
                    loss = - loss * torch.rand(1).cuda()
                    
                loss.backward()
                optimizer.step()
                
                rewd_mean  += reward.sum(-1).detach()
                prof_mean += profit.sum(-1).detach()
        
            tqdm.write(f"Epoch #{ epoch }: Ave. profit { ( prof_mean / args.steps ).item() } Ave. reward { ( rewd_mean / args.steps ).item() }")
            
        torch.save(net.state_dict(), sd_filename)
    else:
        net.load_state_dict(torch.load(sd_filename))
        
    return net
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    parser.add_argument("--profit_discount", default=0.98, type=float, help="Discount of profit computation")
    
    # Trader
    parser.add_argument("--output_daily", default=False, type=bool, help="Whether or not only predict daily action")
    parser.add_argument("--force_retrain", default=False, type=bool, help="Whether or not forcely retrain the agent")
    parser.add_argument("--consider_tradability", default=False, type=bool, help="Whether or not feed tradability into the network")
    parser.add_argument("--seq_encode_mode", default="delta", type=str,
                        help="Mode of encoding price sequence: ['deri-1', 'delta', 'gru']")
    
    # E2E Trading
    parser.add_argument("--initial_waiting", default=50, type=int, help="Days in the beginning without trading")
    parser.add_argument("--seq_stripe", default=False, type=bool, help="Whether or not use seq_len as action stripe size")
    
    # Data
    parser.add_argument("--seq_len", default=16, type=int, help="Length of sequence")
    parser.add_argument("--real_data_path", default="data/data.csv", type=str, help="Path of data")
    parser.add_argument("--data_path", default="data/data_gen.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--epoch", default=40, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--steps", default=256, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", type=str, help="Device of computation")
    
    # Debug
    parser.add_argument("--seq_file_path", default="data/seq_gen.csv", type=str, help="Path of generated sequence file")
    parser.add_argument("--summary_log_dir", default="runs/", type=str, help="Log directory for Tensorboard Summary Writer")
    args = parser.parse_args()
    
    return args

def get_trader(assets, cost_trans, seq_len, consider_tradability=False, output_daily=False, seq_mode='delta', device='cuda'):
    args = parse_arguments()
    
    args.assets = assets
    args.cost_trans = cost_trans
    args.output_daily = output_daily
    args.seq_encode_mode = seq_mode
    args.seq_len = seq_len
    args.consider_tradability = consider_tradability
    args.device = device
    
    # writer = SummaryWriter(args.summary_log_dir)
    return get(args)


if __name__ == '__main__':
    args = parse_arguments()
    
    args.seq_stripe = False
    args.force_retrain = False
    
    writer=SummaryWriter(args.summary_log_dir)
    net = get(args, writer)
    
    with torch.no_grad():
        e2e_run(args, net, plot_trade=True, writer=writer)
    