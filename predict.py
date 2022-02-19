"""
Prediction of prices
Author: Cai Zhou
"""

import statistics
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import acf
# from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm

# load data
from pandas.core.algorithms import diff

btc_data_path = "data/BCHAIN-MKPRU_refine.csv"
gold_data_path = "data/LBMA-GOLD_refine.csv"

data = pd.read_csv("data/data.csv")

btc_data = pd.read_csv(btc_data_path).rename(columns={'Date': 'date', 'Value': 'btc'})
gold_data = pd.read_csv(gold_data_path).rename(columns={'Date': 'date', 'USD': 'gold'})

btc_pd = pd.Series(btc_data['btc'].values, index=btc_data['date'])
gold_pd = pd.Series(data['gold_inter'].values, index=data['date'])

btc_np = data['btc'].values
gold_np = data['gold_inter'].values

btc_diff_1 = btc_pd.diff(1)
gold_diff_1 = gold_pd.diff(1)
btc_diff_2 = diff(arr=btc_np, n=2)
gold_diff_2 = diff(arr=gold_np, n=2)


# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)

def main_VARIMA():

    plt.figure(0)
    space = 23
    train_len = 100
    # df = pd.DataFrame(data['gold_inter'])
    df = np.zeros([1826, 2])
    df[:, 0] = btc_np
    df[:, 1] = gold_np
    df = pd.DataFrame(df)
    for i in range(6):
        print(i)
        if i*space + train_len + space < 1826:
            train = df[i*space:i*space+train_len]
            test = df[i*space+train_len:i*space+train_len+space]
        else:
            break

        # model = pm.auto_arima(df.values, start_p=1, start_q=1,
        #                       information_criterion='aic',
        #                       test='adf',  # use adftest to find optimal 'd'
        #                       max_p=200, max_q=200,  # maximum p and q
        #                       m=1,  # frequency of series
        #                       d=None,  # let model determine 'd'
        #                       seasonal=False,  # No Seasonality
        #                       start_P=0,
        #                       D=0,
        #                       trace=True,
        #                       error_action='ignore',
        #                       suppress_warnings=True,
        #                       stepwise=True)
        model = VARMAX(train, order=(30, 1))
        model_fit = model.fit()
        # model_fit.forecast()
        # print(model_fit.summary())

        # Forecast
        n_periods = space
        fc = model_fit.forecast(steps=n_periods, return_conf_int=True)
        # result = model_fit.get_forecast(n_periods, alpha=0.05)  # 95% conf
        # fc = result.predicted_mean
        # conf = result.conf_int(0.05)
        # se = result.se_mean
        # index_of_fc = np.arange(len(df.values), len(df.values) + n_periods)

        # make series for plotting purpose
        # Make as pandas series
        plt.subplot(211)
        btc_series = pd.Series(fc[:, 0], index=test.index[:, 0])
        plt.plot(btc_series, c='r')
        plt.subplot(212)
        gold_series = pd.Series(fc[:, 1], index=test.index[:, 1])
        # lower_series = pd.Series(conf[:, 0], index=test.index)
        # upper_series = pd.Series(conf[:, 1], index=test.index)

        plt.plot(gold_series, c='r')

    # plt.plot(train, label='training')
    btc_train_data = df[:, 0]
    gold_train_data = df[:, 1]
    plt.subplot(211)
    plt.plot(btc_train_data, label='Bitcoin_actual', c='b')
    plt.subplot(212)
    plt.plot(gold_train_data, label='Gold_actual', c='b')
    # plt.plot(test, label='actual')
    # plt.fill_between(lower_series.index, lower_series, upper_series,
    #                  color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    # plt.xlim(0, 200)
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('./btc_gold_VARMA')

    # plt.figure(1)
    # plt.plot(btc_diff_1)
    # plt.title('btc_diff_1')
    # # plt.show()
    # plt.savefig('./btc_diff_1')
    # plt.figure(2)
    # plt.plot(gold_diff_1)
    # plt.title('gold_diff_1')
    # # plt.show()
    # plt.savefig('./gold_diff_1')
    # plt.figure(3)
    # plt.plot(btc_diff_2)
    # plt.title('btc_diff_2')
    # # plt.show()
    # plt.savefig('./btc_diff_2')
    # plt.figure(4)
    # plt.plot(gold_diff_2)
    # plt.title('gold_diff_2')
    # # plt.show()
    # plt.savefig('./gold_diff_2')


outputpath = "./Predicted_Data.csv"


def main():
    plt.figure(0)
    space = 23
    train_len = 100
    df = pd.DataFrame(data['btc'])
    for i in range(76):
        print(i)
        if i*space + train_len + space < 1826:
            train = df[i*space:i*space+train_len]
            test = df[i*space+train_len:i*space+train_len+space]
        else:
            break

        # model = pm.auto_arima(df.values, start_p=1, start_q=1,
        #                       information_criterion='aic',
        #                       test='adf',  # use adftest to find optimal 'd'
        #                       max_p=200, max_q=200,  # maximum p and q
        #                       m=1,  # frequency of series
        #                       d=None,  # let model determine 'd'
        #                       seasonal=False,  # No Seasonality
        #                       start_P=0,
        #                       D=0,
        #                       trace=True,
        #                       error_action='ignore',
        #                       suppress_warnings=True,
        #                       stepwise=True)
        model = ARIMA(train, order=(30, 1, 15))
        model_fit = model.fit()
        # model_fit.forecast()
        # print(model_fit.summary())

        # Forecast
        n_periods = space
        fc = model_fit.forecast(steps=n_periods, return_conf_int=True)
        fc_ = pd.DataFrame({'btc_23': fc})
        fc_.to_csv('./btc_23.csv', mode='a', sep=',', header=False, index=True, encoding='gbk')
        # result = model_fit.get_forecast(n_periods, alpha=0.05)  # 95% conf
        # fc = result.predicted_mean
        # conf = result.conf_int(0.05)
        # se = result.se_mean
        # index_of_fc = np.arange(len(df.values), len(df.values) + n_periods)

        # make series for plotting purpose
        # Make as pandas series
        fc_series = pd.Series(fc, index=test.index)
        plt.plot(fc_series, c='r')

        # lower_series = pd.Series(conf[:, 0], index=test.index)
        # upper_series = pd.Series(conf[:, 1], index=test.index)

    train_data = df[:]
    plt.plot(train_data, label='actual', c='b')

    plt.title('Forecast vs Actuals')
    # plt.xlim(0, 200)
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('./fig_ARIMA')


def main_HWES():
    plt.figure(0)
    space = 15
    train_len = 100
    df = pd.DataFrame(data['btc'])
    for i in range(365):
        print(i)
        if i * space + train_len + space < 1826:
            train = df[i * space:i * space + train_len]
            test = df[i * space + train_len:i * space + train_len + space]
        else:
            break

        # Forecast
        model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False).fit()
        fc = model.forecast(space)
        fc_ = pd.DataFrame({'btc_HWES': fc})
        fc_.to_csv('./btc_HWES.csv', mode='a', sep=',', header=False, index=True, encoding='gbk')
        plt.plot(list(np.arange(i*space+train_len, i*space+space+train_len, 1)), list(fc), c='r')

    train_data = df[:]
    plt.plot(train_data, label='actual', c='b')

    plt.title('Forecast vs Actuals')
    # plt.xlim(0, 200)
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('./fig_HWES')


if __name__ == '__main__':
    main_HWES()



