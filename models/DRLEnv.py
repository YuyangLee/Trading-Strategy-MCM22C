import numpy as np
import csv
import torch

from utils.data import seq_slide_select


ALPHA_GOLD = 0.01
ALPHA_BTC = 0.02
DATE_NUM = 1826


class Env():
    def __init__(self, args, costs, prices, tradability, seq_len, mode='e2e', device='cuda'):
        self.args = args
        self.assets = args.assets
        
        # B x (num_assets + 1)
        # Index 0 - Balance of cash
        # Else    - Balance of assets
        # self.portfolio = torch.zeros([args.batch_size, num_days + 1, len(args.assets) + 1], device=device)
        # self.portfolio[:, 0, 0] = args.initial_cash
        self.portfolio = torch.zeros([args.batch_size, len(args.assets) + 1], device=device)
        self.portfolio[:, 0] = args.initial_cash
        
        self.costs = torch.from_numpy(np.asarray(costs)).to(device)
        self.seq_len = seq_len
        self.prices = prices
        self.tradability = tradability
        self.mode = mode
        
        self.device = device
        
    def step(self, step, portfolio, sell, buy):
        """
        Calculate effect of day step (in-out in day (step - 1))
        """
        num_assets = sell.shape[-1]
        if self.mode == 'train_from_real':
            prices = self.train_prices
            tradab = self.train_tradability
        elif self.mode == 'e2e':
            prices = self.prices
            tradab = self.tradability
        else:
            raise NotImplementedError()

        
        #for i in range(self.seq_len - 1):
        # Sell
        # assets_trade = portfolio[:, 1:] * sell[:, i]
        assets_trade = portfolio[:, 1:] * sell
        new_cash = portfolio[:, 0] + (assets_trade * tradab[:, step] * (1 - self.costs) * prices[:, step]).sum(-1)
        
        # Buy
        cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy
        # cash_trade = new_cash.unsqueeze(-1).tile((1, num_assets+1)) * buy[:, i]
        new_assets = cash_trade[:, 1:] * tradab[:, step] * (1 - self.costs) / prices[:, step] + portfolio[:, 1:] * (1 - sell)
        # new_assets = cash_trade[:, 1:] * tradab[:, step] * (1 - self.costs) / prices[:, step] + portfolio[:, 1:] * (1 - sell[:, i])
        
        portfolio = torch.concat([
            cash_trade[:, :1],
            new_assets
        ], dim=-1)
        
        new_value = portfolio[:, 0] + (portfolio[:, 1:] * prices[:, step + 1]).sum(-1)
        
        # i += 1
        # ret_prices = prices[:, step+i:step+i+self.seq_len]
        # ret_tradab = tradab[:, step+i:step+i+self.seq_len]
        ret_prices = prices[:, step:step+self.seq_len]
        ret_tradab = tradab[:, step:step+self.seq_len]
        
        return portfolio, new_value, ret_prices, ret_tradab

    def reset(self, **args):
        
        # B x (num_assets + 1)
        # B x deq_len x num_assets 
        if self.mode == 'train_from_real':
            self.train_prices = seq_slide_select(self.prices, self.seq_len + args['episode_len'], False)
            self.train_tradability = seq_slide_select(self.tradability, self.seq_len + args['episode_len'], False)
            
            initial_state = np.asarray([1000., 0.5, 1.])
            portfolio = torch.from_numpy(initial_state).to(self.device).unsqueeze(0).tile((self.args.batch_size, 1)).float()
            portfolio = portfolio * torch.rand(size=portfolio.shape, device=self.device) * 2
            
            prices = self.train_prices[:, 0:self.seq_len]
            tradability = self.train_tradability[:, 0:self.seq_len]
            
        
        elif self.mode == 'e2e':
            prices, tradability = self.prices[:, 0:self.seq_len], self.tradability[:, 0:self.seq_len]
            initial_state = np.asarray([1000., 0., 0.])
            portfolio = torch.from_numpy(initial_state).to(self.device).unsqueeze(0).tile((self.args.batch_size, 1)).float()
            
        elif self.mode == 'trade':
            pass
        
        else:
            raise NotImplementedError()
            
        return portfolio, prices, tradability
