import numpy as np
import csv
import torch


ALPHA_GOLD = 0.01
ALPHA_BTC = 0.02
DATE_NUM = 1826


class Env():
    def __init__(self, args, assets, initial_cash, alphas, prices, tradability, device='cuda'):
        self.device = device
        # date = 1
        
        self.args = args
        self.assets = assets
        self.alphas = alphas
        
        # B x (num_assets + 1)
        # Index 0 - Balance of cash
        # Else    - Balance of assets
        self.portfolio = torch.zeros([args.batch_size, len(assets) + 1], device=device)
        self.portfolio[:, 0] = initial_cash
        
        self.discounts = torch.from_numpy(np.asarray(1 - alphas)).to(device)
        
        self.prices = prices
        self.tradability = tradability
        
    def value(self, prices):
        return 

    def step(self, action, date):
        with open(self.args.data_path, 'r') as f:
            # date_info = list(csv.reader(f))[date]

            file = list(csv.reader(f))
            date_info = file[date]
            price_btc = float(date_info[1])
            gold_tradable = bool(date_info[4])
            # if gold_tradable: price_gold = list(csv.reader(f))[date][2]
            price_gold = float(date_info[3])

            if date > 1: 
                # last_date_info = list(csv.reader(f))[date-1]
                last_date_info = file[date-1]
                last_price_btc = float(last_date_info[1])
                last_price_gold = float(last_date_info[3])
                last_value = self.btc * last_price_btc + self.gold * last_price_gold + self.money
            else: 
                last_value = 1000        
        
        # Purchasing assets
        delta_assets = action * self.tradability[:, date]
        self.portfolio[:, 1:] = self.portfolio + delta_assets
        # with cash balance and costs of transection
        self.portfolio[:, 0] = self.portfolio[:, 0] - (1 + delta_assets) * self.discounts
        
        # self.value = self.btc * price_btc + self.gold * price_gold + self.money
        self.value = self.portfolio[:, 0] + (self.portfolio[:, 1:] * self.prices).sum(-1)

        next_state = torch.concat([
            self.portfolio,
            
        ])
        # next_state = np.array([self.btc, self.gold, self.money, self.value])
        reward = self.value - last_value
        down = True if date == DATE_NUM else False
        info = {}

        return next_state, r, down, info

    def reset(self):
        initial_state = np.array([1000., 0., 0.])
        return torch.from_numpy(initial_state).to(self.device).unsqueeze(0).tile((self.args.batch_size, 1))
