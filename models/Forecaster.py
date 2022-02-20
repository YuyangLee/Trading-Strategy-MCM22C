import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf


class Forecaster():
    def __init__(self, data, device='cuda'):
        # seq_len x ...
        self.data = data
        self.device = device
    
    def forecast(self, step, len, mode='gt', padding=True):
        """
        Forecast a sequence with data indexing step as the head.
        
        E.g. data = [10., 11., 12.,], step = 2, len=3
        => forecast [12., 13., 14.] 

        Args:
            `step`: Index of the last-known data  (i.e. forecasted data index [step : step+len])
            `len`: Length of forecasted sequence
            `mode`: Mode of forecast. Available: ['gt', 'stat', 'seq2seq2']
            `padding`: Padding if the forecasted sequence is not long enough, only used in 'gt' mode

        Returns:
            ground truth sequence (for next-step computation) and predicted sequence (for sampling action)
        """
        truth = self.data[step:step+len].clone()
        if padding and truth.shape[0] < len:
            truth = torch.concat([truth, truth[-1:].tile([len - truth.shape[0]] + [1] * (truth.dim() - 1))], dim=0)
        
        if mode == 'gt':
            forecast = truth.clone()
            
        elif mode == 'stat':
            # TODO: Implementation by C Zhou
            # forecast = ...
            train_btc  = pd.DataFrame(self.data[step - 50 : step+1, 0].to('cpu'))
            train_gold = pd.DataFrame(self.data[step - 50 : step+1, 1].to('cpu'))
            
            model_fit_btc = ARIMA(train_btc, order=(5, 2, 3)).fit()
            model_fit_gold = ARIMA(train_gold, order=(5, 2, 3)).fit()
            fc_btc = torch.tensor(np.array(model_fit_btc.forecast(steps=len-1))).to(self.device)
            fc_btc = torch.concat([self.data[step, 0].unsqueeze(0), fc_btc], dim=-1)
            fc_gold = torch.tensor(np.array(model_fit_gold.forecast(steps=len-1))).to(self.device)
            fc_gold = torch.concat([self.data[step, 1].unsqueeze(0), fc_gold], dim=-1)
            forecast = torch.concat([fc_btc.unsqueeze(0), fc_gold.unsqueeze(0)], dim=0)
            
        elif mode == 'seq2seq':
            pass
        
        return forecast, truth
            