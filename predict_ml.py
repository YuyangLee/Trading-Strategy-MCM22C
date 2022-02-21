# %%
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from utils.data import *
from torch.utils.tensorboard import SummaryWriter
from models.Forecaster import LSTMForecaster, LSTM_FC

# %%
data = get_data("data/data_gen.csv", 'cuda', False)[0][:, :1] / 100

writer = SummaryWriter()

# %%
# train_idx = torch.randint(0, data.shape[0] - train_len, [1])[0]
# test_idx = torch.randint(0, data.shape[0] - test_len, [1])[0]

bs_len = 50
fc_len = 16
batch_size = 2

train_seq = data[:1200+fc_len]
valid_seq  = data[1200:1500+fc_len]
test_seq  = data[1500:]

# %%
forecaster = LSTM_FC(1, fc_len, 1)

# %%
plt.figure(figsize=(30,30))

plt.subplot(311)
plt.title("Train")
plt.plot(train_seq.cpu())

plt.subplot(312)
plt.title("Valid")
plt.plot(valid_seq.cpu())

plt.subplot(313)
plt.title("Test")
plt.plot(test_seq.cpu())

plt.show()

# %%
x_train, y_train = [], []
for i in range(bs_len, train_seq.shape[0] - fc_len):
    x_train.append(train_seq[i - bs_len:i])
    y_train.append(train_seq[i: i+fc_len])
    
x_valid, y_valid = [], []
for i in range(bs_len, valid_seq.shape[0] - fc_len):
    x_valid.append(valid_seq[i - bs_len:i])
    y_valid.append(valid_seq[i: i+fc_len])
    
x_test, y_test = [], []
for i in range(bs_len, test_seq.shape[0] - fc_len):
    x_test.append(test_seq[i - bs_len:i])
    y_test.append(test_seq[i: i+fc_len])

print(f"Train: { len(x_train) } Valid: { len(x_valid) } Test: { len(x_test) }")

# %%
loss_fn = nn.MSELoss()
# for i in trange(epoch):
for i in range(20):
    optimizer = torch.optim.SGD(forecaster.parameters(), lr=(1e-4) * (0.95**i))
    for j in trange(0, len(x_train) - batch_size):
        xx, yy = torch.stack(x_train[j:j+batch_size]), torch.stack(y_train[j:j+batch_size])
        # xx, yy = x_train[i], y_train[i]
        # xx = xx.permute([1, 0, 2])
        # yy = yy.permute([1, 0, 2])
        pred = forecaster(xx, future=fc_len, y=yy)
        
        optimizer.zero_grad()
        loss = loss_fn(yy.reshape((-1, 2)), pred.reshape((-1, 2)))
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("Loss", loss.item(), i * 20 + j)
        
        if j % 100 == 0:
            tqdm.write(f"Epoch #{ i } step #{ j } loss = { loss }")
    
    with torch.no_grad():
        j = torch.randint(0, len(x_test) - batch_size, (1,))
        xx_test, yy_test = torch.stack(x_test[j:j+batch_size]), torch.stack(y_test[j:j+batch_size])
        pred = forecaster(xx_test, future=fc_len)
        loss = loss_fn(yy.reshape((-1, 2)), pred.reshape((-1, 2)))
        tqdm.write(f"Test loss = { loss }")

# %%

plt.figure(figsize=(16,9))
plt.plot(np.arange(bs_len), xx[0].detach().cpu(), 'blue')
plt.plot(np.arange(bs_len, bs_len+fc_len), yy[0].detach().cpu(), 'green')
plt.plot(np.arange(bs_len, bs_len+fc_len), pred[0].detach().cpu(), 'red')

plt.savefig("fig.pdf")
