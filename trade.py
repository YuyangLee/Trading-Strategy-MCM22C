import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn

from utils.utils import *

data_path = "data/data.csv"

if __name__ == '__main__':
    with torch.no_grad():
        pd.read_csv(data_path)
