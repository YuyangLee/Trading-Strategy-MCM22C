import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from get_trader import sample_tradability
from models.DRLEnv import Env
from models.Policy import Seq2SeqPolicy


class DPG(object):
    def __init__(self, batch_size, num_assets, memory_capacity, num_replay=32, seq_len=16, lr=1e-3, device='cuda'):
        self.num_assets = num_assets
        self.state_len = num_assets + 1 + seq_len * num_assets
        self.batch_size = batch_size
        self.num_replay = num_replay
        
        self.policy = Seq2SeqPolicy(num_assets, seq_len, consider_tradability=False, output_daily=True, seq_mode='delta', device=device)
        self.policy.train()
        
        # Memory of DDPG
        self.memory_counter = 0 # 统计更新记忆次数
        self.memory_capacity = memory_capacity
        self.memory = torch.zeros((batch_size, memory_capacity, self.state_len * 2 + num_assets + 2), device=device) # 记忆库初始化
        # self.memory = torch.zeros((mem_cap))

        # Optimizers
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        
        self.device = device
        
    def lr_decay(self, gamma=0.999):
        self.lr = self.lr * gamma
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr)

    def choose_action(self, portfolio, price_seq, tradability):
        """
        Choose action from state

        Args:
            `portfolio`:  B x (num_assets + 1)
            `price_seq`:  B x seq_len x num_assets
            `tradability`: B x seq_len x num_assets
        """
        return self.policy(portfolio, price_seq, tradability)

def main(args, writer):
    num_assets = len(args.assets)
    
    if not os.path.isfile(args.data_path):
        raise NotImplementedError()
    print(f"Reading data from file { args.data_path }")
    
    data = pd.read_csv(args.data_path)
    
    # num_days x num_assets with prices in USD
    df = pd.DataFrame(data=data, columns=['btc', 'gold_inter'])
    prices_data = torch.from_numpy(df.to_numpy()).float().to(args.device).unsqueeze(0).tile((args.batch_size, 1, 1))
    # num_days x num_assets with {0., 1.}
    tradability_data = torch.from_numpy(pd.DataFrame(data=data, columns=['btc_tradable', 'gold_tradable']).to_numpy()).float().to(args.device).unsqueeze(0).tile((args.batch_size, 1, 1))
    num_days_data = prices_data.shape[1]
    
    print(f"========== Data Loaded ==========")
    print(df.describe())
    print(f"Totally { data.shape[0] } days of trade, with { torch.from_numpy(data['gold_tradable'].to_numpy() == False).int().sum().item() } unavailable for gold.")
    print(f"========== Data Loaded ==========")
    
    dpg = DPG(batch_size=args.batch_size, num_assets=len(args.assets), seq_len=args.seq_len, memory_capacity=args.memory_capacity, device=args.device)
    
    test_env = Env(args=args,
                   seq_len=args.seq_len,
                   costs=[args.cost_trans[asset] for asset in args.assets],
                   prices=prices_data,
                   tradability=tradability_data,
                   mode='e2e')
    portfolio_test, prices_test, tradability_test = test_env.reset()
    
    for episode in trange(args.num_episode):        
        env = Env(
            args=args,
            seq_len=args.seq_len,
            costs=[args.cost_trans[asset] for asset in args.assets],
            prices=prices_data,
            tradability=tradability_data,
            # tradability=sample_tradability(args.batch_size, args.seq_len, num_assets, indices=[-1], device=args.device),
            mode='train_from_real'
        )
        
        if args.mode == 'train_from_real':
            train_kwargs = {
                "episode_len": args.seq_len * 2
            }
            step_lo = 0
            step_hi = train_kwargs['episode_len']
            
        else:
            step_lo, step_hi = args.episode_steps_range[0], args.episode_steps_range[1]
            
        portfolio, prices, tradability = env.reset(**train_kwargs)
        reward = 0

        for step in range(step_lo, step_hi):
            ret_discount = 0.999
            
            buy, sell = dpg.choose_action(portfolio,
                                          prices,
                                          tradability)
                        
            portfolio, ret, prices, tradability = env.step(step, portfolio, buy, sell)
            
            # Optimize
            reward += ret * ret_discount
            ret_discount *= ret_discount
            
        # if episode % 200 == 0:
        #     with torch.no_grad():
        #         for step_test in range(0, num_days_data - args.seq_len):
        #             buy_test, sell_test = dpg.choose_action(portfolio_test,
        #                                         prices_test,
        #                                         tradability_test)
                                
        #             portfolio_test, ret_test, prices_test, tradability_test = test_env.step(step_test, portfolio_test, buy_test, sell_test)
        #     tqdm.write(f"Test: Total profit { ret_test }")
            
        reward = reward.sum(-1)
        
        dpg.optimizer.zero_grad()
        loss = - reward
        loss.backward()
        dpg.optimizer.step()
        dpg.lr_decay()
            
        # if episode > 20:
        #     # writer.add_scalars(f"Batch #0, Episode #{ episode }/{ args.num_episode }", {
        #     writer.add_scalars(f"Batch #0, Episode #{ episode }/{ args.num_episode }", {
        #         "Cash": portfolio[0, 0].detach().cpu(),   
        #         "BTC": portfolio[0, 1].detach().cpu(),   
        #         "Gold": portfolio[0, 2].detach().cpu(),
        #         "Value": reward.detach().cpu()
        #     }, step)
            
        tqdm.write(f"Episode: { episode },  Reward: { reward.item() }")

def parse_arguments(agile=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", default="train_from_real", type=str, help="Running mode")
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    
    # Experience Episodes
    parser.add_argument("--num_episode", default=2000, type=int, help="Length of an episode")
    parser.add_argument("--episode_steps_range", default=[0, 1826], type=list, help="Range of steps in an episode")
    parser.add_argument("--memory_capacity", default=1826 * 50, type=int, help="Capacity of memory")
    
    # Actor
    parser.add_argument("--seq_len", default=32, type=int, help="Len of price sequence as part of the state")
    parser.add_argument("--action_var", default=1.0, type=float, help="Var of action noises, will decay through steps")
    parser.add_argument("--var_decay_rate", default=0.9999, type=float, help="Decay rate of Var of action noises")
    
    # Data
    parser.add_argument("--data_path", default="data/data.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", type=str, help="Device of computation")
    
    args = parser.parse_args()
    
    if agile:
        args.batch_size = 2
        args.memory_capacity = 1826 * 20
        
    return args


if __name__ == '__main__':
    args = parse_arguments(agile=False)
    writer = SummaryWriter("runs/")
    main(args, writer)
