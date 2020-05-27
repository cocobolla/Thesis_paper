# Author: Chanju Park (Cocobolla)
# Date: 2020.05.25

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import os
import warnings

import pairs
import backtest
from pair_statistics import Statistics

# System Setting
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%4f' % x)
warnings.filterwarnings('ignore')


def main():
    ###########################
    #       Data Loading      #
    ###########################
    root_path = './pickle'
    ticker_pkl = 'kse_18_20_ticker.pkl'
    sector_pkl = 'kse_sector.pkl'
    price_pkl = 'kse_18_20.pkl'
    ticker_pkl = os.path.join(root_path, ticker_pkl)
    sector_pkl = os.path.join(root_path, sector_pkl)
    price_pkl = os.path.join(root_path, price_pkl)

    ticker_series = pd.read_pickle(ticker_pkl)
    sector_df = pd.read_pickle(sector_pkl)
    df = pd.read_pickle(price_pkl)

    ###########################
    #   Data Pre-processing   #
    ###########################
    df = df.dropna(axis=1)
    df_price = df['수정주가(원)']

    form_str_date = datetime.datetime(2018, 1, 1)
    form_end_date = datetime.datetime(2018, 12, 31)
    trading_str_date = datetime.datetime(2019, 1, 1)
    trading_end_date = datetime.datetime(2019, 6, 30)

    formation_close = df_price[form_str_date:form_end_date]
    trading_close = df_price[trading_str_date:trading_end_date]

    ###########################
    #      Finding Pairs      #
    ###########################
    if not os.path.isfile('pickle/pairs3.pkl'):
        df_pairs = pairs.find_pairs(formation_close)
        # df_pairs.to_pickle('pickle/pairs.pkl')
    else:
        print('Get Pairs from Pickle...')
        df_pairs = pd.read_pickle('pickle/pairs.pkl')

    ###########################
    #    Trading with Pairs   #
    ###########################
    ret_df = pd.DataFrame()
    for i in range(len(df_pairs)):
        print(i)
        trading_history, _ = backtest.get_pair_returns(df_pairs.loc[i, :], formation_close, trading_close)
        ret_df[i] = trading_history['quantity_neutral']

    ###########################
    #        Statistics       #
    ###########################
    Statistics(ret_df).print_statistics()


def pairs_trading(formation_price, trading_price):
    ###########################
    #      Finding Pairs      #
    ###########################
    if not os.path.isfile('pickle/pairs3.pkl'):
        df_pairs = pairs.find_pairs(formation_price)
        # df_pairs.to_pickle('pickle/pairs.pkl')
    else:
        print('Get Pairs from Pickle...')
        df_pairs = pd.read_pickle('pickle/pairs.pkl')

    ###########################
    #    Trading with Pairs   #
    ###########################
    ret_df = pd.DataFrame()
    for i in range(len(df_pairs)):
        print(i)
        trading_history, _ = backtest.get_pair_returns(df_pairs.loc[i, :], formation_price, trading_price)
        ret_df[i] = trading_history['quantity_neutral']

    ###########################
    #        Statistics       #
    ##########################
    trading_stat = Statistics(ret_df)
    return trading_stat


func_o = '__main__'
func_t = '__main__1'

if __name__ == func_t:
    ###########################
    #       Data Loading      #
    ###########################
    root_path = './pickle'
    ticker_pkl = 'kse_18_20_ticker.pkl'
    sector_pkl = 'kse_sector.pkl'
    # price_pkl = 'kse_18_20.pkl'
    price_pkl = 'kse_11_20.pkl'
    ticker_pkl = os.path.join(root_path, ticker_pkl)
    sector_pkl = os.path.join(root_path, sector_pkl)
    price_pkl = os.path.join(root_path, price_pkl)

    ticker_series = pd.read_pickle(ticker_pkl)
    sector_df = pd.read_pickle(sector_pkl)
    df = pd.read_pickle(price_pkl)

    ###########################
    #   Data Pre-processing   #
    ###########################
    # df = df.dropna(axis=1)
    # df_price = df['수정주가(원)']

    # Set Formation & Trading Period
    year_list = df.index.year.unique()
    # Formation period: 12 months, Trading period: 6 months
    trading_period = relativedelta(months=6)
    formation_period = relativedelta(months=12)
    total_period = trading_period + formation_period

    formation_str_list = ([datetime.datetime(y, 1, 1) for y in year_list if
                           datetime.datetime(y, 1, 1) + total_period < df.index[-1]] +
                          [datetime.datetime(y, 7, 1) for y in year_list if
                           datetime.datetime(y, 7, 1) + total_period < df.index[-1]])
    formation_str_list = sorted(formation_str_list)
    trading_str_list = [x + formation_period for x in formation_str_list]

    formation_end_list = [x + formation_period + relativedelta(days=-1) for x in formation_str_list]
    trading_end_list = [x + trading_period + relativedelta(days=-1) for x in trading_str_list]

    stat_list = []
    for fs, fe, ts, te in zip(formation_str_list, formation_end_list, trading_str_list, trading_end_list):
        print('{} ~ {}'.format(fs, te))
        df_price = df.loc[fs:te, '수정주가(원)'].copy()
        df_price = df_price.dropna(axis=1)
        print('# of Stocks: {}'.format(len(df_price.columns)))

        formation_close = df_price[fs:fe]
        trading_close = df_price[ts:te]
        stat = pairs_trading(formation_close, trading_close)
        stat.print_statistics()
        stat_list.append(stat)
    pd.to_pickle(stat_list, './pickle/stat_pickle_SC.pkl')


if __name__ == func_o:
    ###########################
    #       Data Loading      #
    ###########################
    root_path = './pickle'
    ticker_pkl = 'kse_18_20_ticker.pkl'
    sector_pkl = 'kse_sector.pkl'
    # price_pkl = 'kse_18_20.pkl'
    price_pkl = 'kse_11_20.pkl'
    ticker_pkl = os.path.join(root_path, ticker_pkl)
    sector_pkl = os.path.join(root_path, sector_pkl)
    price_pkl = os.path.join(root_path, price_pkl)

    ticker_series = pd.read_pickle(ticker_pkl)
    sector_df = pd.read_pickle(sector_pkl)
    df = pd.read_pickle(price_pkl)

    ###########################
    #   Data Pre-processing   #
    ###########################
    # df = df.dropna(axis=1)
    # df_price = df['수정주가(원)']

    # Set Formation & Trading Period
    # fs = datetime.datetime(2018, 1, 1)
    # fe = datetime.datetime(2018, 12, 31)
    # ts = datetime.datetime(2019, 1, 1)
    # te = datetime.datetime(2019, 6, 30)
    fs = datetime.datetime(2018, 7, 1)
    fe = datetime.datetime(2019, 6, 30)
    ts = datetime.datetime(2019, 7, 1)
    te = datetime.datetime(2019, 12, 31)
    # fs = datetime.datetime(2013, 1, 1)
    # fe = datetime.datetime(2013, 12, 31)
    # ts = datetime.datetime(2014, 1, 1)
    # te = datetime.datetime(2014, 6, 30)

    df_price = df.loc[fs:te, '수정주가(원)'].copy()
    df_price = df_price.dropna(axis=1)

    print('{} ~ {}'.format(fs, te))
    print('# of Stocks: {}'.format(len(df_price.columns)))

    formation_close = df_price[fs:fe]
    trading_close = df_price[ts:te]
    stat = pairs_trading(formation_close, trading_close)
    stat.print_statistics()

