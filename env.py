import numpy as np
import csv
import torch


ALPHA_GOLD = 0.01
ALPHA_BTC = 0.02
DATE_NUM = 1826


class Env():
    def __init__(self, args, alphas, prices, tradability, device='cuda'):
        self.device = device
        # date = 1
        
        self.args = args
        self.assets = args.assets
        self.alphas = alphas
        
        num_days = prices.shape[0]
        
        # B x (num_assets + 1)
        # Index 0 - Balance of cash
        # Else    - Balance of assets
        # self.portfolio = torch.zeros([args.batch_size, num_days + 1, len(args.assets) + 1], device=device)
        # self.portfolio[:, 0, 0] = args.initial_cash
        self.portfolio = torch.zeros([args.batch_size, len(args.assets) + 1], device=device)
        self.portfolio[:, 0] = args.initial_cash
        
        self.alphas = torch.from_numpy(np.asarray(alphas)).to(device)
        
        self.prices = prices
        self.tradability = tradability
        
    def step(self, action, step, last_portfolio):
        """
        Calculate effect of day step (in-out in day (step - 1))
        """
        date = step + 1
        # date_info = list(csv.reader(f))[date]

        if date > 1: 
            last_date_prices = self.prices[date - 2]
            # last_date_value  = (self.portfolio[:, date - 1, 1:] * last_date_prices).sum(-1) + self.portfolio[:, date - 1, 0]
            last_date_value  = (last_portfolio[:, 1:] * last_date_prices).sum(-1) + self.portfolio[:, 0]
        else: 
            last_date_value = self.portfolio[:, 0] * 1. + 0.        
        
        # Purchasing assets
        # TODO: Discuss about the input of tradability: whether the forward pass of Actor, or here with action given...
        # self.portfolio[:, date, 1:] = self.portfolio[:, date - 1, 1:] + action * (1 - self.alphas)
        self.portfolio[:, 1:] = last_portfolio[:, 1:] + action * (1 - self.alphas)
        # with cash balance and costs of transection
        # self.portfolio[:, date, 0] = self.portfolio[:, date - 1, 0] - (action * self.prices[date - 1]).sum(-1)
        self.portfolio[:, 0] = last_portfolio[:, 0] - (action * self.prices[date - 1]).sum(-1)
        
        if self.portfolio[:, 0] + 1e-4 < 0:
            print("Wrong")
        # self.value = self.btc * price_btc + self.gold * price_gold + self.money
        value = self.portfolio[:, 0] + (self.portfolio[:, 1:] * self.prices[date - 1]).sum(-1)

        ret = value - last_date_value

        return self.portfolio, value, ret

    def reset(self):
        initial_state = np.array([1000., 0., 0.])
        return torch.from_numpy(initial_state).to(self.device).unsqueeze(0).tile((self.args.batch_size, 1)).float()
