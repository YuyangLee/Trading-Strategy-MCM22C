from thirdparty.dts.deepseries.model.rnn2rnn import RNN2RNN
from thirdparty.dts.deepseries.train import Learner
from thirdparty.dts.deepseries.data import Value, create_seq2seq_data_loader, forward_split
from thirdparty.dts.deepseries.nn import RMSE, MSE
import thirdparty.dts.deepseries.functional as F
import numpy as np
import torch

batch_size = 16
enc_len = 20
dec_len = 20
series_len = 1000

epoch = 100
lr = 0.001

valid_size = 12
test_size = 12

series = np.sin(np.arange(0, series_len)) + np.random.normal(0, 0.1, series_len) + np.log2(np.arange(1, series_len+1))
series = series.reshape(1, 1, -1)

train_idx, valid_idx = forward_split(np.arange(series_len), enc_len=enc_len, valid_size=valid_size+test_size)
valid_idx, test_idx = forward_split(valid_idx, enc_len, test_size)

# mask test, will not be used for calculating mean/std.
mask = np.zeros_like(series).astype(bool)
mask[:, :, test_idx] = False
series, mu, std = F.normalize(series, axis=2, fillna=True, mask=mask)

# create train/valid dataset
train_dl = create_seq2seq_data_loader(series[:, :, train_idx], enc_len, dec_len, sampling_rate=0.1,
                                      batch_size=batch_size, seq_last=True, device='cuda')
valid_dl = create_seq2seq_data_loader(series[:, :, valid_idx], enc_len, dec_len,
                                      batch_size=batch_size, seq_last=True, device='cuda')

# define model
wave = RNN2RNN(enc_len, dec_len, 10).cuda()
# wave = Wave2Wave(target_size=1, num_layers=6, num_blocks=1, dropout=0.1, loss_fn=RMSE())
opt = torch.optim.Adam(wave.parameters(), lr=lr)

# train model
wave_learner = Learner(wave, opt, root_dir="data/predict/dts/wave")
wave_learner.fit(max_epochs=epoch, train_dl=train_dl, valid_dl=valid_dl, early_stopping=True, patient=16)

# load best model
wave_learner.load(wave_learner.best_epoch)

# predict and show result
import matplotlib.pyplot as plt
wave_preds = wave_learner.model.predict(torch.tensor(series[:, :, test_idx[:-12]]).float().cuda(), 12).cpu().numpy().reshape(-1)

plt.plot(series[:, :, -48:-12].reshape(-1))
plt.plot(np.arange(36, 48), wave_preds, label="wave2wave preds")
plt.plot(np.arange(36, 48), series[:, :, test_idx[-12:]].reshape(-1), label="target")
plt.legend()