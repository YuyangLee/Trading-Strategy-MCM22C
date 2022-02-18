import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange


class SeqEncoder(nn.Modules):
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
        

class MLPPolicy(nn.Modules):
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

