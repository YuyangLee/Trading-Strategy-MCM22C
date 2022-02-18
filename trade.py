import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import torch

from utils.utils import *

data_path = "data/data.csv"

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Assets
    parser.add_argument("--assets", default=['btc', 'gold'], type=list, help="Assets available for trading")
    parser.add_argument("--cost_trans", default={'btc': 0.02, 'gold': 0.01}, type=dict, help="Cost of transection of given assets")
    parser.add_argument("--initial_cash", default=1000.0, type=float, help="Default amount of cash")
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    with torch.no_grad():
        pd.read_csv(data_path)

