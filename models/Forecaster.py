import torch
import numpy as np

class Forecaster():
    def __init__(self, data):
        # seq_len x ...
        self.data = data
    
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
            pass
        
        elif mode == 'seq2seq':
            pass
        
        return truth, forecast
            