import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import Env


EPISODE = 200
STEP = 1826
BATCH_SIZE = 32
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01  # 参数软更新的更新系数
MEMORY_CAPACITY = 1826 * 50
N_ACTIONS = 2
N_STATES = 4


# 搭建策略网络
class Actor(nn.Module):
    def __init__(self, ):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 256)
        self.fc1.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(256, N_ACTIONS)
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
    def __init__(self, ):
        super(Critic, self).__init__()

        self.fc_s = nn.Linear(N_STATES, 256)
        self.fc_s.weight.data.normal_(0, 0.1)

        self.fc_a = nn.Linear(N_ACTIONS, 256)
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
    def __init__(self, ):
        self.actor_eval, self.actor_target = Actor(), Actor()
        self.critic_eval, self.critic_target = Critic(), Critic()
        
        self.memory_counter = 0 # 统计更新记忆次数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + N_ACTIONS + 1)) # 记忆库初始化

        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_A)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_C)
        self.loss_func = nn.MSELoss()   # 损失函数

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)    # 升维
        action = self.actor_eval(x)[0].detach()
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
            last_s_ = torch.FloatTensor(last_transition[-N_STATES:])
            
            last_btc = last_s_.numpy()[0]
            last_gold = last_s_.numpy()[1]
            last_money = last_s_.numpy()[2]
            last_value = last_s_.numpy()[3]

            return last_btc, last_gold, last_money, last_value

    def learn(self):
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
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.FloatTensor(b_memory[:, N_STATES: N_STATES + N_ACTIONS])
        b_r = torch.FloatTensor(b_memory[:, -N_STATES - 1: -N_STATES])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

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


def main():
    ddpg = DDPG()
    var = 0.5     # 动作噪声的方差，会随着时间衰减

    print('\nCollecting experience...')

    for episode in range(EPISODE):        
        env = Env() # 生成环境
        s = env.reset()
        ep_r = 0    # 累计奖励值初始化

        for step in range(STEP):
            date = step + 1
            
            # 读取价格
            with open('data.csv', 'r') as f:
                date_info = list(csv.reader(f))[date]
                price_btc = float(date_info[1])
                gold_tradable = bool(date_info[4])
                # if gold_tradable: price_gold = list(csv.reader(f))[date][2]
                price_gold = float(date_info[3])

            last_btc, last_gold, last_money, last_value = ddpg.read_last_state(date)

            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var, size=2), [-last_btc, -last_gold], [last_money / price_btc, last_money / price_gold])    # 动作选择时添加噪声
            s_, r, done, info = env.step(a, date)
            ddpg.store_transition(s, a, r, s_)
            ep_r += r

            if ddpg.memory_counter > MEMORY_CAPACITY:   # 记忆库满后开始更新神经网络参数
                var *= 0.99999      # 动作噪声衰减
                ddpg.learn()
                if done:
                    print('Episode: ', episode, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)

            if done:
                break

            s = s_  # 状态更新            


if __name__ == '__main__':
    main()
