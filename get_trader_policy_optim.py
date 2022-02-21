import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from models.Forecaster import Forecaster
from models.Policy import Seq2SeqPolicy
from utils.data import *

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
    seq_len = prices_gt.shape[1]
    num_assets = prices_gt.shape[2]
    
    init_value = pf[:, 0] + (pf[:, 1:] * prices_gt[:, 0]).sum(-1).detach()
    
    # B x s x 2, B x s x 3
    sell, buy = net_fn(pf, prices_fc, tradability)
    
    # reg_vec = torch.concat([sell, buy], dim=-1).reshape((-1, num_assets * 2 + 1))
    # reg = (reg_vec * (1 - reg_vec)).sum(-1)
    
    profit = torch.zeros([batch_size], device=pf.device)
    new_pf = pf.clone()
    
    reg = 0
    
    for date in range(seq_len - 1):
        # Sell
        assets_sell_amount = new_pf[:, 1:] * sell[:, date] * tradability[:, date]
        new_cash = new_pf[:, 0] + (assets_sell_amount * (1 - costs) * prices_gt[:, date]).sum(-1)
        
        # Buy
        cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, date]
        assets_buy_prices = cash_trade[:, 1:] * tradability[:, date]
        new_assets = pf[:, 1:] - assets_sell_amount + assets_buy_prices * (1 - costs) / prices_gt[:, date]
        cash_remain = new_cash - assets_buy_prices.sum(-1)
        
        new_pf = torch.concat([
            cash_remain.unsqueeze(-1),
            new_assets
        ], dim=-1)
        
        new_value = new_pf[:, 0] + (new_pf[:, 1:] * prices_gt[:, date + 1]).sum(-1)
        profit = profit + (new_value - init_value) * gamma
        reg = reg + (sell * buy[..., 1:]).sum(-1).sum(-1).sum(-1)
        
    if ret_pf:
        return profit, new_value - init_value, new_value, new_pf, reg
    else:
        return profit, new_value - init_value, reg


def e2e_run(args, net, plot_trade=True, mode='gt', writer=None):
    prices, tradability = get_data(args.real_data_path, args.device)
    num_assets = 2
    num_days = prices.shape[0]
    
    costs = torch.from_numpy(np.asarray([args.cost_trans[asset] for asset in args.assets])).to(args.device).unsqueeze(0)
    
    # Post-padding for tradability, not needed for prices.
    prices = torch.concat([prices, prices[-args.seq_len:]], dim=0).unsqueeze(0)
    tradability = torch.concat([tradability, tradability[-args.seq_len:]], dim=0).unsqueeze(0)
    pf = torch.tensor([1000., 0., 0.]).to(args.device).unsqueeze(0)

    net.eval()
    
    to_plot = pd.DataFrame()
    
    for step in trange(args.initial_waiting, num_days):
        prices_seq = prices[:, step:step+args.seq_len]
        sell, buy = net(pf, prices_seq, tradability[:, step:step+args.seq_len])
        # Sell
        assets_sell_amount = pf[:, 1:] * sell[:, 0] * tradability[:, step]
        new_cash = pf[:, 0] + (assets_sell_amount * (1 - costs) * prices_seq[:, 0]).sum(-1)
        # Buy
        cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, 0]
        assets_buy_prices = cash_trade[:, 1:] * tradability[:, step]
        new_assets = pf[:, 1:] - assets_sell_amount + assets_buy_prices * (1 - costs) / prices_seq[:, 0]
        cash_remain = new_cash - assets_buy_prices.sum(-1)
        
        pf = torch.concat([
            cash_remain.unsqueeze(-1),
            new_assets
        ], dim=-1)
        
        new_value = (pf[:, 0] + (pf[:, 1:] * prices_seq[:, 1]).sum(-1))
        
        plot = {
            "cash_balance": cash_trade[0, 0].detach().cpu().item(),
            "btc_balance":  pf[0, 1].detach().cpu().item(),
            "gold_balance": pf[0, 2].detach().cpu().item(),
            "price_btc": prices[0, step, 0].detach().cpu().item(),
            "price_gold": prices[0, step, 1].detach().cpu().item(),
            "value": new_value[0].detach().cpu().item(),
            "sell_btc": assets_sell_amount[0, 0].detach().cpu().item(),
            "sell_gold": assets_sell_amount[0, 0].detach().cpu().item(),
            "sell_btc_ratio": sell[0, 0, 0].detach().cpu().item(),
            "sell_gold_ratio": sell[0, 0, 1].detach().cpu().item(),
            "buy_no_ratio": buy[0, 0, 0].detach().cpu().item(),
            "buy_btc_ratio": buy[0, 0, 1].detach().cpu().item(),
            "buy_gold_ratio": buy[0, 0, 2].detach().cpu().item(),
            "buy_btc": cash_trade[0, 1].detach().cpu().item(),
            "buy_gold": cash_trade[0, 2].detach().cpu().item(),
            "cash_remain": cash_trade[0, 0].detach().cpu().item()
        }
        to_plot = to_plot.append(plot, ignore_index=True)
        tqdm.write(f"Step { step }: value = { new_value[0].item() }, pf = { pf.detach().cpu().numpy() }")
    new_value = new_value[0]
    
    # new_value = new_value.detach().cpu().item()
    print(f"""
========== Evaluation ==========
Value: { new_value.item() } with profit { new_value.item() - 1000. }
================================
          """)
    
    return new_value.item(), to_plot
        
def get(args, writer=None, sd_filename=None): 
    net = Seq2SeqPolicy(len(args.assets), args.seq_len, consider_tradability=args.consider_tradability, seq_mode=args.seq_encode_mode, output_daily=args.output_daily, device=args.device)
    if sd_filename is None:
        sd_filename = os.path.join("data", "trader", f"{args.seq_len}_{ args.seq_encode_mode }_{ 'cons' if args.consider_tradability else 'ignr' }_{ args.epoch }.pt")
    
    if args.force_retrain or not os.path.isfile(sd_filename):
        prices, tradability = get_data(args.data_path, args.device, sample_tradability=False, untradability_assets=[-1])
        prices = prices / 10 + 200 + 500 * torch.rand((1)).to(args.device)
        
        num_days = prices.shape[0]
        num_assets = len(args.assets)
        
        net.train()
        prof_traj = []
        costs = torch.from_numpy(np.asarray([args.cost_trans[asset] for asset in args.assets])).to(args.device).unsqueeze(0)
        for epoch in range(args.epoch):
            optimizer = torch.optim.Adam(params=net.parameters(), lr=(1e-3)*(0.95**epoch))
            for step in trange(args.steps):
                init_pf = sample_init_pf(args.batch_size, num_assets, ranges=[[0., 20000.], [0., 10.], [0., 200.]], device='cuda')

                seq_start_idx = torch.randint(0, num_days - args.seq_len, [args.batch_size], device=args.device)
                price_seq = torch.zeros([0, args.seq_len, num_assets], device=args.device)
                tdblt_seq = torch.zeros([0, args.seq_len, num_assets], device=args.device)
                
                for batch in range(args.batch_size):
                    price_seq = torch.concat([price_seq, prices[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len].clone().unsqueeze(0)], dim=0)
                    tdblt_seq = torch.concat([tdblt_seq, tradability[seq_start_idx[batch]:seq_start_idx[batch] + args.seq_len].clone().unsqueeze(0)], dim=0)
                
                # price_seq = prices[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
                # tdblt_seq = tradability[step:step+args.seq_len].unsqueeze(0).tile((args.batch_size, 1, 1))
                
                reward, acc_gain, reg = run_network(step, init_pf, price_seq, price_seq, tdblt_seq, net, costs=costs, gamma=args.profit_discount, plot_trade=False, writer=writer)
                
                optimizer.zero_grad()
                
                # if epoch < 20:
                # loss = - (reward + acc_gain).sum(-1)
                # else:
                loss = - ((reward + acc_gain).sum(-1) + 100 * (acc_gain.mean() / acc_gain.std()))
                # loss = 10.0 * reg - ((profit).sum(-1) + (profit.mean() / profit.std()))
                          
                epsilon = torch.rand(1)
                if epsilon < 0.4 * np.exp(-0.5 * epoch):
                    loss = - loss * torch.rand(1).cuda()
                    
                loss.backward()
                optimizer.step()
                
            tqdm.write(f"Epoch #{ epoch }: Reward { reward.mean().item() }")
            
        torch.save(net.state_dict(), sd_filename)
    else:
        net.load_state_dict(torch.load(sd_filename))
        
    # print(prof_traj)
    return net
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    parser.add_argument("--profit_discount", default=0.95, type=float, help="Discount of profit computation")
    
    # Trader
    parser.add_argument("--output_daily", default=False, type=bool, help="Whether or not only predict daily action")
    parser.add_argument("--force_retrain", default=False, type=bool, help="Whether or not forcely retrain the agent")
    parser.add_argument("--consider_tradability", default=False, type=bool, help="Whether or not feed tradability into the network")
    parser.add_argument("--seq_encode_mode", default="deri-1", type=str,
                        help="Mode of encoding price sequence: ['deri-1', 'delta', 'gru']")
    
    # E2E Trading
    parser.add_argument("--initial_waiting", default=0, type=int, help="Days in the beginning without trading")
    parser.add_argument("--seq_stripe", default=False, type=bool, help="Whether or not use seq_len as action stripe size")
    
    # Data
    parser.add_argument("--seq_len", default=16, type=int, help="Length of sequence")
    parser.add_argument("--real_data_path", default="data/data.csv", type=str, help="Path of data")
    parser.add_argument("--data_path", default="data/data_gen.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--epoch", default=50, type=int, help="Epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--steps", default=512, type=int, help="Batch size")
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

    # Validate:
    # args.real_data_path = "data/data.csv"
    # args.file_path = 'data/data.csv'
    # args.seq_file_path = 'data/data.csv'
    
    net = get(args, writer)
    mode = 'gt'
    profit, to_plot = e2e_run(args, net, plot_trade=True, mode=mode, writer=writer)
    to_plot.to_csv(os.path.join("data", "trading", f"{args.seq_len}_{ args.seq_encode_mode }_{ mode }_{ 'cons' if args.consider_tradability else 'ignr' }.csv"))
    sns.relplot(y='value', data=to_plot)
    