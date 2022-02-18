import numpy as np
import csv


ALPHA_GOLD = 0.01
ALPHA_BTC = 0.02
DATE_NUM = 1826


class Env():
    def __init__(self):
        # date = 1
        self.btc = 0        # 个
        self.gold = 0       # 盎司
        self.money = 1000   # 美元
        self.value = 1000   # 美元

    def step(self, a, date):
        with open('data.csv', 'r') as f:
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
        
        self.btc += a[0]
        self.gold += a[1]
        self.money = self.money - a[0] * price_btc - a[1] * price_gold
        self.money = self.money - abs(a[0]) * price_btc * ALPHA_BTC - abs(a[1]) * price_gold * ALPHA_GOLD   # 扣除手续费
        self.value = self.btc * price_btc + self.gold * price_gold + self.money

        s_ = np.array([self.btc, self.gold, self.money, self.value])
        r = self.value - last_value
        down = True if date == DATE_NUM else False
        info = {}

        return s_, r, down, info

    def reset(self):
        return np.array([0, 0, 1000, 1000])
