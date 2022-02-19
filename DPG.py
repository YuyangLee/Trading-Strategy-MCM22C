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

from env import Env
from utils import get_data

EPISODE = 200
STEP = 1826
BATCH_SIZE = 4
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01

class Policy(nn.Module):
    """
    The Policy Network
    """
    def __init__(self, num_assets, seq_len=16, device='cuda'):
        super(Policy, self).__init__()

        # Cash and balances of assets, with seq_len days of prices in the future
        state_len = num_assets + 1 + seq_len * num_assets  
        self.num_assets = num_assets
        
        self.fc1 = nn.Linear(state_len, 32).to(device)
        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(32, num_assets).to(device)
        self.out.weight.data.normal_(0.,0.1)
        
        self.device = device
        
    def action_bound(self, portfolio, trade, prices):
        cash = portfolio[..., 0]
        assets = portfolio[..., 1:]

    def forward(self, state, prices, tradability=None, noise_var=None):
        output = self.fc1(state)
        output = F.relu(output)
        output = self.out(output)
        output = torch.tanh(output)
        if noise_var is not None:
            output = output + torch.normal(0, noise_var, size=output.shape, device=output.device)
        if tradability is not None:
            output = output * tradability
        # output = output * self.action_bound
        
        ouput = self.action_bound(state[..., :self.num_assets + 1], output, prices)
        return output


class DPG(object):
    def __init__(self, batch_size, num_assets, memory_capacity, num_replay=32, seq_len=16, device='cuda'):
        self.num_assets = num_assets
        self.state_len = num_assets + 1 + seq_len * num_assets
        self.batch_size = batch_size
        self.num_replay = num_replay
        
        self.policy = Policy(num_assets, args.seq_len)
        self.policy.train()
        
        # Memory of DDPG
        self.memory_counter = 0 # 统计更新记忆次数
        self.memory_capacity = memory_capacity
        self.memory = torch.zeros((batch_size, memory_capacity, self.state_len * 2 + num_assets + 2), device=device) # 记忆库初始化
        # self.memory = torch.zeros((mem_cap))

        # Optimizers
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR_A)
        self.loss_func = nn.MSELoss()
        
        self.device = device

    def choose_action(self, state, tradability, action_var):
        """
        Choose action from state

        Args:
            `state`: state input, B x state_len
        """
        action = self.policy(state, tradability, action_var)
        return action

    def learn(self):
        # Updating target network
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')

        # sample_index = torch.random.choice(self.memory_capacity, self.memory_capacity)
        sample_index = torch.randint(0, self.memory_capacity, [self.batch_size * self.num_replay], device=self.device)
        b_memory = self.memory.reshape((-1, self.memory.shape[-1]))
        b_memory = b_memory[sample_index]
        # (B x BATCH_SIZE) x 10
        
        b_s  = b_memory[:, :self.state_len]
        b_a  = b_memory[:, self.state_len:self.state_len + self.num_assets]
        b_r  = b_memory[:, -self.state_len - 1: -self.state_len]
        b_s_ = b_memory[:, -self.state_len:]

        # Get action and compute its Q
        a = self.actor_eval(b_s)
        q = self.critic_eval(b_s, a)
        a_loss = - torch.mean(q)

        # Update evaluate networks
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # 假想做出下一步动作并打分
        a_target = self.actor_target(b_s_)
        q_next   = self.critic_target(b_s_, a_target)
        q_target = b_r + GAMMA * q_next
        q_eval   = self.critic_eval(b_s, b_a)
        c_loss   = self.loss_func(q_eval, q_target)

        # Update target network
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()


def main(args, writer):
    prices, tradability = get_data(args)
    
    dpg = DPG(batch_size=args.batch_size, num_assets=len(args.assets), memory_capacity=args.memory_capacity, device=args.device)
    
    for episode in trange(args.episode_len):        
        env = Env(
            args=args,
            alphas = [args.cost_trans[asset] for asset in args.assets],
            prices=prices, tradability=tradability
        )
        
        state = env.reset(args.seq_len)
        ep_r = 0

        for step in range(args.episode_steps_range[0], args.episode_steps_range[1]):
            ret_discount = 0.999
            
            date = step + 1
            portfolio = state[:, :len(args.assets) + 1]
            
            action = dpg.choose_action(state, tradability[date - 1], args.action_var)
            action = torch.clamp(action, -portfolio[:, 1:], portfolio[:, 0].unsqueeze(-1).tile((1, len(args.assets))) / prices[step] / 2)
            
            state, value, ret = env.step(action, step, portfolio, args.seq_len)
            
            # if state[:, 0] + 1e-3 < 0:
            #     print("Wrong")
                
            # Optimize
            ep_r += ret * ret_discount
            ret_discount *= ret_discount
        
            if step % 100 == 0:
                args.action_var *= args.var_decay_rate
            
        dpg.optimizer.zero_grad()
        loss = - ep_r
        loss.backward()
        dpg.optimizer.step()

            
        if episode > 20:
            # writer.add_scalars(f"Batch #0, Episode #{ episode }/{ args.episode_len }", {
            writer.add_scalars(f"Batch #0, Episode #{ episode }/{ args.episode_len }", {
                "Cash": state[0, 0].detach().cpu(),   
                "BTC": state[0, 1].detach().cpu(),   
                "Gold": state[0, 2].detach().cpu(),
                "Value": value[0].detach().cpu()
            }, step)
            
        tqdm.write(f"Episode: { episode },  Reward: { ep_r.item() }, Explore: { args.action_var }")

def parse_arguments(agile=False):
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    
    # Experience Episodes
    parser.add_argument("--episode_len", default=2000, type=int, help="Length of an episode")
    parser.add_argument("--episode_steps_range", default=[0, 1826], type=list, help="Range of steps in an episode")
    parser.add_argument("--memory_capacity", default=1826 * 50, type=int, help="Capacity of memory")
    
    # Actor
    parser.add_argument("--seq_len", default=256, type=int, help="Len of price sequence as part of the state")
    parser.add_argument("--action_var", default=1.0, type=float, help="Var of action noises, will decay through steps")
    parser.add_argument("--var_decay_rate", default=0.9999, type=float, help="Decay rate of Var of action noises")
    
    # Data
    parser.add_argument("--data_path", default="data/data.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--device", default="cuda", type=str, help="Device of computation")
    
    args = parser.parse_args()
    
    if agile:
        args.batch_size = 1
        args.memory_capacity = 1826 * 20
        
    return args


if __name__ == '__main__':
    args = parse_arguments(agile=True)
    writer = SummaryWriter("runs/")
    main(args, writer)
