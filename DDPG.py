import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from env import Env

EPISODE = 200
STEP = 1826
BATCH_SIZE = 32
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 1826 * 50
state_len = 4


# 搭建策略网络
class Actor(nn.Module):
    def __init__(self, num_assets):
        super(Actor, self).__init__()

        state_len = num_assets + 1  # Cash and balances of assets
        self.fc1 = nn.Linear(state_len, 256)
        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(256, num_assets)
        self.out.weight.data.normal_(0.,0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        # actions_value = x * self.action_bound   # 限定action范围
        actions_value = x
        return actions_value


# 搭建价值网络
class Critic(nn.Module):
    def __init__(self, num_assets):
        super(Critic, self).__init__()

        state_len = num_assets + 1  # Cash and balances of assets
        self.fc_s = nn.Linear(state_len, 256)
        self.fc_s.weight.data.normal_(0, 0.1)

        self.fc_a = nn.Linear(num_assets, 256)
        self.fc_a.weight.data.normal_(0.,0.1)

        self.out = nn.Linear(256, 1)
        self.out.weight.data.normal_(0.,0.1)

    def forward(self, s, a):
        s = self.fc_s(s)
        a = self.fc_a(a)
        n = F.relu(s+a)
        q_value = self.out(n)
        return q_value


# 编写DDPG算法
class DDPG(object):
    def __init__(self, num_assets):
        self.num_assets = num_assets
        state_len = num_assets + 1
        self.actor_eval, self.actor_target = Actor(num_assets), Actor(num_assets)
        self.critic_eval, self.critic_target = Critic(), Critic()
        
        self.memory_counter = 0 # 统计更新记忆次数
        self.memory = np.zeros((MEMORY_CAPACITY, state_len * 2 + num_assets + 1)) # 记忆库初始化
        # self.memory = torch.zeros((mem_cap))

        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        """
        Choose action from state

        Args:
            `state`: state input, B x state_len
        """
        action = self.actor_eval(state).detach()
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_)) # 将数据压缩成一维数组（横向拼接）
        index = self.memory_counter % MEMORY_CAPACITY   # 计算待储存的下标（超出容量自动循环覆盖原有记忆）
        self.memory[index, :] = transition  # 记忆更新
        self.memory_counter += 1

    def read_last_state(self, date):
        if date == 1: 
            return 0, 0, 1000, 1000
        else: 
            index = (self.memory_counter - 1) % MEMORY_CAPACITY
            last_transition = self.memory[index, :]
            last_s_ = torch.FloatTensor(last_transition[-state_len:])
            
            last_btc = last_s_.numpy()[0]
            last_gold = last_s_.numpy()[1]
            last_money = last_s_.numpy()[2]
            last_value = last_s_.numpy()[3]

            return last_btc, last_gold, last_money, last_value

    def learn(self):
        state_len = self.num_assets + 1
        # target网络参数软更新
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')

        # 随机抽取部分记忆并提取对应部分
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :state_len])
        b_a = torch.FloatTensor(b_memory[:, state_len: state_len + self.num_assets])
        b_r = torch.FloatTensor(b_memory[:, -state_len - 1: -state_len])
        b_s_ = torch.FloatTensor(b_memory[:, -state_len:])

        # 做出动作并进行打分
        a = self.actor_eval(b_s)
        q = self.critic_eval(b_s, a)
        a_loss = -torch.mean(q)

        # 策略网络参数更新
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # 假想做出下一步动作并打分
        a_target = self.actor_target(b_s_)
        q_next = self.critic_target(b_s_, a_target)
        q_target = b_r + GAMMA * q_next
        q_eval = self.critic_eval(b_s, b_a)
        c_loss = self.loss_func(q_eval, q_target)

        # 价值网络参数更新
        self.critic_optimizer.zero_grad()
        c_loss.backward()
        self.critic_optimizer.step()


def main(args):
    if os.path.isfile(args.data_path):
        raise NotImplementedError()
    print(f"Reading data from file { args.data_path }")
    
    data = pd.read_csv(args.data_path)
    
    # num_days x num_assets with prices in USD
    prices = torch.from_numpy(pd.DataFrame(data=data, columns=['btc', 'gold_inter']).to_numpy()).float().to(args.device)
    # num_days x num_assets with {0., 1.}
    tradability = torch.from_numpy(pd.DataFrame(data=data, columns=['btc_tradable', 'gold_tradable']).to_numpy()).float().to(args.device)
    
    print(f"========== Data Loaded ==========")
    prices.describe()
    print(f"Totally { data.shape[0] } days of trade, with { torch.from_numpy(data['gold_tradable'].to_numpy() == False).int().sum().item() } unavailable for gold.")
    print(f"========== Data Loaded ==========")
    
    ddpg = DDPG(num_assets=len(args.assets))
    
    print(f"Collecting experience from episodes { args.episode_range }")

    for episode in trange(args.episode_len):        
        env = Env(
            batch_size=args.batch_size,
            assets=args.batch_size,
            initial_cash=args.initial_cash,
            alphas = [args.cost_trans[asset] for asset in args.assets],
            prices=prices, tradability=tradability
        )
        
        state = env.reset()
        ep_r = 0

        for step in range(args.episode_steps_range[0], args.episode_steps_range[1]):
            last_btc, last_gold, last_money, last_value = ddpg.read_last_state(step)
            
            action = ddpg.choose_action(state)
            action = torch.clamp(torch.normal(action, args.action_var, size=2), [-last_btc, -last_gold], [last_money / price_btc, last_money / price_gold])    # 动作选择时添加噪声
            next_state, reward, done, info = env.step(action, step)
            ddpg.store_transition(state, action, r, next_state)
            ep_r += reward

            if ddpg.memory_counter > MEMORY_CAPACITY:   # 记忆库满后开始更新神经网络参数
                var *= 0.99999      # 动作噪声衰减
                ddpg.learn()
                if done:
                    print('Episode: ', episode, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)

            if done:
                break

            s = next_state  # 状态更新            

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Portfolio
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    
    # Experience Episodes
    parser.add_argument("--episode_len", default=200, type=int, help="Length of an episode")
    parser.add_argument("--episode_steps_range", default=[0, 1826], type=list, help="Range of steps in an episode")
    
    # Actor
    parser.add_argument("--action_var", default=0.5, type=float, help="Var of action noises, will decay through steps")
    
    # Data
    parser.add_argument("--data_path", default="data/data.csv", type=str, help="Path of data")
    
    # Computation
    parser.add_argument("--device", default="cuda", type=str, help="Device of computation")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
