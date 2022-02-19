import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange


class Seq2SeqPolicy(nn.Module):
    def __init__(self, num_assets, seq_len, consider_tradability=True, seq_hidden_size=16, seq_mode='delta', device='cuda'):
        super(Seq2SeqPolicy, self).__init__()
        
        self.num_assets = num_assets
        self.seq_len = seq_len
        
        # Portfolio, prices
        
        self.seq_mode = seq_mode
        
        if seq_mode == 'delta':
            self.input_len  = 1 + num_assets * seq_len + num_assets
            self.seq_enc = lambda x: (x - x[:, 0].unsqueeze(1).expand(x.shape)).reshape([x.shape[0], -1]).detach()
        elif seq_mode == 'deri-1':
            self.input_len  = 1 + num_assets * seq_len + num_assets - num_assets 
            self.seq_enc = lambda x: (x[:, 1:] - x[:, :-1]).reshape([x.shape[0], -1]).detach()
        elif seq_mode == 'gru':
            self.input_len  = 1 + num_assets + seq_hidden_size
            self.gru = nn.GRU(input_size=2, hidden_size=seq_hidden_size, batch_first=True).to(device)
            self.seq_enc = lambda x: self.gru(x)[0][:, -1]
        else:
            raise NotImplementedError()
        
        self.consider_tradability = consider_tradability
        
        if consider_tradability:
            self.input_len += num_assets * seq_len
            
        self.output_len = seq_len * (num_assets * 2 + 1)
        
        self.policy = nn.Sequential(
            nn.Linear(self.input_len, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.output_len),
            nn.Sigmoid()
        ).to(device)
        
        self.device = device
        
    def forward(self, init_portfolio, prices_seq, tradability=None):
        """
        Forward pass of Seq2seqPolicy Network

        Args:
            `init_portfolio`: B x (num_assets + 1)
            `prices_seq`: B x seq_len x num_assets
            `tradability`: B x seq_len x num_assets
        """
        seq_code = self.seq_enc(prices_seq)
        
        input = torch.concat([
            init_portfolio,
            seq_code
        ], dim=-1)
        
        if self.consider_tradability:
            input = torch.concat([
                input,
                tradability.reshape((tradability.shape[0], -1))
            ], dim=-1)
            
        action = self.policy(input).view([input.shape[0], self.seq_len, -1])
        
        sell = action[..., :self.num_assets]
        buy  = F.normalize(action[..., self.num_assets:], p=1, dim=-1)

        return sell, buy

class SeqEncoder(nn.Module):
    def __init__(self, num_assets, output_len):
        SeqEncoder(MLPPolicy, self).__init__()
        
        self.num_assets = num_assets
        self.output_len = output_len
        
        self.l1 = nn.GRUCell(input_size=num_assets, hidden_size=10)
        self.l2 = nn.Linear(16, 3)

    def forward(self, input):
        input = torch.relu(self.l1(input))
        input = torch.relu(self.l2(self.output_len))
        
        return input        
        

class MLPPolicy(nn.Module):
    def __init__(self, output_len, num_assets, seq_feature_len):
        super(MLPPolicy, self).__init__()
        
        self.output_len = output_len
        self.seq_feature_len = seq_feature_len
        
        self.seq_enc = SeqEncoder(num_assets=num_assets, output_len=seq_feature_len)
        
        self.l1 = nn.Linear(seq_feature_len, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, output_len)
        
    def forward(self, input):
        """
        Forward pass.

        Args:
            `input`: B x seq_len x 2
        """
        # Encoding input: B
        input = self.seq_enc(input)
        input = torch.relu(self.l1(input))
        input = torch.relu(self.l2(input))
        input = torch.sigmoid(self.l3(input))

        return input

def run_trader(args):
    policy_net = MLPPolicy(output_len=2, num_assets=len(args.assets), seq_feature_len=16)
    pass
