import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter

from env import Env

EPISODE = 200
STEP = 1826
BATCH_SIZE = 4
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
state_len = 4


class Actor(nn.Module):
    """
    The Actor Network
    """
    def __init__(self, num_assets, device='cuda'):
        super(Actor, self).__init__()

        state_len = num_assets + 1  # Cash and balances of assets
        self.fc1 = nn.Linear(state_len, 256).to(device)
        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(256, num_assets).to(device)
        self.out.weight.data.normal_(0.,0.1)
        
        self.device = device

    def forward(self, state, tradability=None, noise_var=None):
        output = self.fc1(state)
        output = F.relu(output)
        output = self.out(output)
        output = torch.tanh(output)
        if noise_var is not None:
            output = output + torch.normal(0, noise_var, size=output.shape, device=output.device)
        if tradability is not None:
            output = output * tradability
        # output = output * self.action_bound
        return output


class Critic(nn.Module):
    """
    The Critic Network
    """
    def __init__(self, num_assets, device='cuda'):
        super(Critic, self).__init__()

        state_len = num_assets + 1  # Cash and balances of assets
        self.fc_s = nn.Linear(state_len, 256).to(device)
        self.fc_s.weight.data.normal_(0, 0.1)

        self.fc_a = nn.Linear(num_assets, 256).to(device)
        self.fc_a.weight.data.normal_(0.,0.1)

        self.out = nn.Linear(256, 1).to(device)
        self.out.weight.data.normal_(0.,0.1)
        
        self.device = device

    def forward(self, s, a):
        s = self.fc_s(s)
        a = self.fc_a(a)
        n = F.relu(s+a)
        q_value = self.out(n)
        return q_value


class DDPG(object):
    def __init__(self, batch_size, num_assets, memory_capacity, device='cuda'):
        self.num_assets = num_assets
        state_len = num_assets + 1
        self.batch_size = batch_size
        
        self.actor_eval, self.actor_target = Actor(num_assets), Actor(num_assets)
        self.critic_eval, self.critic_target = Critic(num_assets), Critic(num_assets)
        
        # Memory of DDPGe
        self.memory_counter = 0 # 统计更新记忆次数
        self.memory_capacity = memory_capacity
        self.memory = torch.zeros((batch_size, memory_capacity, state_len * 2 + num_assets + 2), device=device) # 记忆库初始化
        # self.memory = torch.zeros((mem_cap))

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        self.loss_func = nn.MSELoss()
        
        self.device = device

    def choose_action(self, state, tradability, action_var):
        """
        Choose action from state

        Args:
            `state`: state input, B x state_len
        """
        action = self.actor_eval(state, tradability, action_var).detach()
        return action

    def store_transition(self, state, action, value, reward, next_state):
        """
        Storage a state in the memory

        Args:
            `state`: B x state_len
            `action`: B x num_assets
            `reward`: B
            `next_state`: B x state_len
        """
        transition = torch.concat((state, action, value.unsqueeze(-1), reward.unsqueeze(-1), next_state), dim=-1)
        self.memory[:, self.memory_counter % self.memory_capacity] = transition
        self.memory_counter += 1

    def read_last_state(self, date):
        if date == 1: 
            return torch.from_numpy(np.asarray([1000, 0, 0])).unsqueeze(0).tile((self.batch_size, 1)).to(self.device), 1000.
        else: 
            last_transition = self.memory[:, (self.memory_counter - 1) % self.memory_capacity, :]
            # last_portfolio = last_transition[:, :self.num_assets + 1]
            last_portfolio = last_transition[:, -(self.num_assets + 1):]
            last_value = last_transition[:, self.num_assets * 2 + 1]

            return last_portfolio, last_value

    def learn(self):
        state_len = self.num_assets + 1
        # Updating target network
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')

        # sample_index = torch.random.choice(self.memory_capacity, self.memory_capacity)
        sample_index = torch.randint(0, self.memory_capacity, [self.batch_size * BATCH_SIZE], device=self.device)
        b_memory = self.memory.reshape((-1, self.memory.shape[-1]))
        b_memory = b_memory[sample_index]
        # (B x BATCH_SIZE) x 10
        
        b_s  = b_memory[:, :state_len]
        b_a  = b_memory[:, state_len:state_len + self.num_assets]
        b_r  = b_memory[:, -state_len - 1: -state_len]
        b_s_ = b_memory[:, -state_len:]

        # Get action and compute its Q
        a = self.actor_eval(b_s)
        q = self.critic_eval(b_s, a)
        a_loss = -torch.mean(q)

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
    if not os.path.isfile(args.data_path):
        raise NotImplementedError()
    print(f"Reading data from file { args.data_path }")
    
    data = pd.read_csv(args.data_path)
    
    # num_days x num_assets with prices in USD
    df = pd.DataFrame(data=data, columns=['btc', 'gold_inter'])
    prices = torch.from_numpy(df.to_numpy()).float().to(args.device)
    # num_days x num_assets with {0., 1.}
    tradability = torch.from_numpy(pd.DataFrame(data=data, columns=['btc_tradable', 'gold_tradable']).to_numpy()).float().to(args.device)
    
    print(f"========== Data Loaded ==========")
    print(df.describe())
    print(f"Totally { data.shape[0] } days of trade, with { torch.from_numpy(data['gold_tradable'].to_numpy() == False).int().sum().item() } unavailable for gold.")
    print(f"========== Data Loaded ==========")
    
    ddpg = DDPG(batch_size=args.batch_size, num_assets=len(args.assets), memory_capacity=args.memory_capacity, device=args.device)
    
    print(f"Collecting experience from { args.episode_len } episodes")

    for episode in trange(args.episode_len):        
        env = Env(
            args=args,
            alphas = [args.cost_trans[asset] for asset in args.assets],
            prices=prices, tradability=tradability
        )
        
        state = env.reset()
        ep_r = 0

        for step in range(args.episode_steps_range[0], args.episode_steps_range[1]):
            date = step + 1
            last_portfolio, _ = ddpg.read_last_state(date)
            
            action = ddpg.choose_action(state, tradability[date - 1], args.action_var)
            action = torch.clamp(action, -last_portfolio[:, 1:], last_portfolio[:, 0].unsqueeze(-1).tile((1, len(args.assets))) / prices[step] / 2)
            next_state, value, ret = env.step(action, step, last_portfolio)
            
            if next_state[:, 0] + 1e-4 < 0:
                print("Wrong")
            ddpg.store_transition(state, action, value, ret, next_state)
            ep_r += ret

            if ddpg.memory_counter > args.memory_capacity:
                # tqdm.write("Learning from experiences...")
                args.action_var *= args.var_decay_rate
                ddpg.learn()
                
            if episode > 20:
                # writer.add_scalars(f"Batch #0, Episode #{ episode }/{ args.episode_len }", {
                writer.add_scalars(f"Batch #0, Episode #{ episode }/{ args.episode_len }", {
                    "Cash": next_state[0, 0].detach().cpu(),   
                    "BTC": next_state[0, 1].detach().cpu(),   
                    "Gold": next_state[0, 2].detach().cpu(),
                    "Value": value[0].detach().cpu()
                }, step)
                
            state = next_state
            
        tqdm.write(f"Episode: { episode },  Reward: { ep_r }, Explore: { args.action_var }")

def parse_arguments(agile=False):
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    
    # Experience Episodes
    parser.add_argument("--episode_len", default=200, type=int, help="Length of an episode")
    parser.add_argument("--episode_steps_range", default=[0, 1826], type=list, help="Range of steps in an episode")
    parser.add_argument("--memory_capacity", default=1826 * 50, type=int, help="Capacity of memory")
    
    # Actor
    parser.add_argument("--action_var", default=0.5, type=float, help="Var of action noises, will decay through steps")
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
