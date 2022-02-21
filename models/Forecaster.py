import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import acf

import numpy.random as random

class LSTM_FC(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, device='cuda'):
    super(LSTM_FC, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.lstm = nn.LSTMCell(self.input_size, self.hidden_size).to(device)
    self.linear = nn.Linear(self.hidden_size, self.output_size).to(device)
    self.device = device

  def forward(self, input, future=0, y=None):
    scaler = input.max()
    input = input / scaler
    outputs = []

    #reset the state of LSTM
    #the state is kept till the end of the sequence
    h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
    c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(self.device)

    # for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
    for i in range(input.shape[1]):
      h_t, c_t = self.lstm(input[:, 1], (h_t, c_t))
      output = self.linear(h_t)
      # outputs += [output]

    for i in range(future): #teacher forcing
      if y is not None:
        output = y[:,i]
      h_t, c_t = self.lstm(output, (h_t,c_t))
      output = self.linear(h_t)
      outputs += [output]
    outputs = torch.stack(outputs,1).squeeze(2)
    return outputs * scaler

# class LSTMForecaster(nn.Module):
class LSTMForecaster(nn.Module):
    def __init__(self, num_assets, fc_len, device='cuda'):
        super(LSTMForecaster, self).__init__()
        
        self.l1 = nn.LSTM(num_assets, 10).to(device)
        self.l2 = nn.Dropout(0.1).to(device)
        self.l3 = nn.Linear(10, fc_len).to(device)
        self.device = device
        
    def forward(self, input):
        output = self.l1(input.permute([1, 0, 2])),
        output = self.l2(output[0])
        output = self.l3(output).permute([1, 0, 2])
        
        return output
        

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
            
        elif mode == 'ema':
            train_btc  = pd.DataFrame(self.data[step - 50 : step+1, 0].to('cpu'))
            train_gold = pd.DataFrame(self.data[step - 50 : step+1, 1].to('cpu'))
            
            model_fit_btc = ExponentialSmoothing(train_btc, trend='add', seasonal=None, damped_trend=False).fit()
            model_fit_gold = ExponentialSmoothing(train_gold, trend='add', seasonal=None, damped_trend=False).fit()
            fc_btc = torch.tensor(np.array(model_fit_btc.forecast(steps=len-1))).to(self.device)
            fc_btc = torch.concat([self.data[step, 0].unsqueeze(0), fc_btc], dim=-1)
            fc_gold = torch.tensor(np.array(model_fit_gold.forecast(steps=len-1))).to(self.device)
            fc_gold = torch.concat([self.data[step, 1].unsqueeze(0), fc_gold], dim=-1)
            forecast = torch.concat([fc_btc.unsqueeze(-1), fc_gold.unsqueeze(-1)], dim=-1)
            
        elif mode == 'stat':
            # ARIMA
            train_btc  = np.log(pd.DataFrame(self.data[step - 50 : step+1, 0].to('cpu')))
            train_gold = np.log(pd.DataFrame(self.data[step - 50 : step+1, 1].to('cpu')))
            
            model_fit_btc = ARIMA(train_btc, order=(15, 1, 8)).fit()
            model_fit_gold = ARIMA(train_gold, order=(15, 1, 8)).fit()
            fc_btc = torch.exp(torch.tensor(np.array(model_fit_btc.forecast(steps=len-1))).to(self.device))
            fc_btc = torch.concat([self.data[step, 0].unsqueeze(0), fc_btc], dim=-1)
            fc_gold = torch.exp(torch.tensor(np.array(model_fit_gold.forecast(steps=len-1))).to(self.device))
            fc_gold = torch.concat([self.data[step, 1].unsqueeze(0), fc_gold], dim=-1)
            forecast = torch.concat([fc_btc.unsqueeze(0), fc_gold.unsqueeze(0)], dim=0)
            
            #VARIMA
            # train = pd.DataFrame(self.data[step - 50 : step + 1].to('cpu'))
            # model = VARMAX(train, order=(8, 2), trend='ct', measurement_error=True)
            # model_fit = model.fit(maxiter=15)
            # fc = model_fit.forecast(steps=len-1)
            # forecast = torch.concat([self.data[step:step+1], torch.from_numpy(np.array(fc)).to(self.device)], dim=0)
            
        elif mode == 'seq2seq':
            pass
        
        return forecast, truth
            