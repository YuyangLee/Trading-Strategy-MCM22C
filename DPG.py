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

from get_trader import get_trader, sample_tradability
from models.DRLEnv import Env
from models.Policy import Seq2SeqPolicy


class DPG(object):
    def __init__(self, args, lr=1e-3, device='cuda'):
        seq_len = args.seq_len

        self.assets = args.assets
        self.num_assets = len(self.assets)
        self.batch_size = args.batch_size

        # self.policy = get_trader(self.assets,
        #                          cost_trans=args.cost_trans,
        #                          consider_tradability=False,
        #                          output_daily=False,
        #                          seq_mode='delta',
        #                          seq_len=args.seq_len,
        #                          device=args.device)
        
        self.policy = Seq2SeqPolicy(self.num_assets, seq_len, consider_tradability=False,
                                    output_daily=False, seq_mode='delta', device=device)
        self.policy.load_state_dict(torch.load(args.sd_path))
        self.policy.train()

        # Optimizers
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.device = args.device

    def lr_decay(self, gamma=0.999):
        self.lr = self.lr * gamma
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=self.lr)
        return self.lr

    def choose_action(self, portfolio, price_seq, tradability):
        """
        Choose action from state

        Args:
            `portfolio`:  B x (num_assets + 1)
            `price_seq`:  B x seq_len x num_assets
            `tradability`: B x seq_len x num_assets
        """
        sell, buy = self.policy(portfolio, price_seq, tradability)
        return sell[:, 0], buy[:, 0]
        # return sell, buy

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
    tradability_data = torch.from_numpy(pd.DataFrame(data=data, columns=['btc_tradable', 'gold_tradable']).to_numpy(
    )).float().to(args.device).unsqueeze(0).tile((args.batch_size, 1, 1))
    num_days_data = prices_data.shape[1]

    print(f"========== Data Loaded ==========")
    print(df.describe())
    print(
        f"Totally { data.shape[0] } days of trade, with { torch.from_numpy(data['gold_tradable'].to_numpy() == False).int().sum().item() } unavailable for gold.")
    print(f"========== Data Loaded ==========")

    dpg = DPG(args)

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

        ret_discount = 0.99
        init_value = (portfolio[:, 0] + (portfolio[:, 1:] * prices[:, 0]).sum(-1)).detach()
        for step in range(step_lo, step_hi):
        # for step in range(step_lo, step_hi, args.seq_len - 1):

            sell, buy = dpg.choose_action(portfolio,
                                          prices,
                                          tradability)

            portfolio, new_value, prices, tradability = env.step(step, portfolio, sell, buy)

            # Optimize

        # if episode % 200 == 0:
        #     with torch.no_grad():
        #         for step_test in range(0, num_days_data - args.seq_len):
        #             buy_test, sell_test = dpg.choose_action(portfolio_test,
        #                                         prices_test,
        #                                         tradability_test)

        #             portfolio_test, ret_test, prices_test, tradability_test = test_env.step(step_test, portfolio_test, buy_test, sell_test)
        #     tqdm.write(f"Test: Total profit { ret_test }")

        reward = (new_value - init_value)

        # dpg.optimizer.zero_grad()
        # loss = - reward.sum(-1) + reward.mean() / reward.std()
        # loss.backward()
        # dpg.optimizer.step()
        lr = dpg.lr_decay()


        tqdm.write(f"Episode: { episode },  Ave. Reward: { reward.mean().item() } Lr: { lr }")
        writer.add_scalars(f"Ave. Reward", {
            "Value": reward.mean().item()
        }, episode)


def parse_arguments(agile=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train_from_real",
                        type=str, help="Running mode")

    # Portfolio
    parser.add_argument(
        "--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={
                        'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0,
                        type=float, help="Default amount of cash")

    # Experience Episodes
    parser.add_argument("--num_episode", default=2000,
                        type=int, help="Length of an episode")
    parser.add_argument("--episode_steps_range",
                        default=[0, 1826], type=list, help="Range of steps in an episode")
    parser.add_argument("--memory_capacity", default=1826 *
                        50, type=int, help="Capacity of memory")

    # Actor
    parser.add_argument("--seq_len", default=16, type=int,
                        help="Len of price sequence as part of the state")
    parser.add_argument("--action_var", default=1.0, type=float,
                        help="Var of action noises, will decay through steps")
    parser.add_argument("--var_decay_rate", default=0.9999,
                        type=float, help="Decay rate of Var of action noises")

    # Data
    
    parser.add_argument("--sd_path", default="data/trader/16_delta_ignr_40.pt",
                        type=str, help="Path of state dict data fro traid trader")
    parser.add_argument("--data_path", default="data/data.csv",
                        type=str, help="Path of data")

    # Computation
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda",
                        type=str, help="Device of computation")

    args = parser.parse_args()

    if agile:
        args.batch_size = 2

    return args


if __name__ == '__main__':
    args = parse_arguments(agile=False)
    writer = SummaryWriter("runs/")
    main(args, writer)
