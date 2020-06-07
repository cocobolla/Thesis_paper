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
import talib

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


def pairs_trading(formation_price, trading_price, ml_list=None):
    ###########################
    #      Finding Pairs      #
    ###########################
    pickle_name = str(formation_price.index[0].date()) + '_' + str(trading_price.index[-1].date())
    if not os.path.isfile('pickle/{}_not.pkl'.format(pickle_name)):
        df_pairs, eig_weights = pairs.find_pairs(formation_price)
        # pd.Series([df_pairs, eig_weights], index=['pairs', 'eig_weights'])\
            # .to_pickle('pickle/{}.pkl'.format(pickle_name))
    else:
        print('Get Pairs from Pickle...')
        pair_pkl = pd.read_pickle('pickle/{}.pkl'.format(pickle_name))
        df_pairs, eig_weights = pair_pkl['pairs'], pair_pkl['eig_weights']

    # Calculate Eigen-Portfolio(Risk Factor)'s Return
    formation_idx = formation_price.index
    trading_idx = trading_price.index
    total_price = formation_price.append(trading_price)
    total_return = total_price.pct_change().dropna(axis=0)
    formation_return = total_return.loc[formation_idx, :]
    trading_return = total_return.loc[trading_idx, :]

    formation_eig_return = pd.DataFrame()
    trading_eig_return = pd.DataFrame()
    for i in range(len(eig_weights)):
        formation_eig_return[i] = formation_return.mul(eig_weights[i], axis=1).sum(axis=1)
        trading_eig_return[i] = trading_return.mul(eig_weights[i], axis=1).sum(axis=1)

    eig_list = [formation_eig_return, trading_eig_return]

    ###########################
    #    Trading with Pairs   #
    ###########################
    # ret_df = pd.DataFrame()
    rest_dict = {}
    for i in range(len(df_pairs)):
        print(i)
        if ml_list is None:
            trading_history, _ = backtest.get_pair_returns(df_pairs.loc[i, :], formation_price, trading_price)
        else:
            trading_history, _ = backtest.get_pair_returns_ml(df_pairs.loc[i, :],
                                                              formation_price, trading_price, ml_list, eig_list)
        rest_dict[i] = trading_history#['quantity_neutral']

    ###########################
    #        Statistics       #
    ###########################
    trading_stat = Statistics(rest_dict)
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
    # price_pkl = 'kse_11_20.pkl'
    price_pkl = 'kse_01_20_volume.pkl'
    ticker_pkl = os.path.join(root_path, ticker_pkl)
    sector_pkl = os.path.join(root_path, sector_pkl)
    price_pkl = os.path.join(root_path, price_pkl)

    ticker_series = pd.read_pickle(ticker_pkl)
    sector_df = pd.read_pickle(sector_pkl)
    df_ori = pd.read_pickle(price_pkl)

    Close = df_ori['수정주가(원)']
    Volume = df_ori['거래량(주)']
    Cap = df_ori['시가총액 (52주 평균)(백만원)']
    assert(Close.isnull().sum()).equals(Volume.isnull().sum())

    ###########################
    #   Data Pre-processing   #
    ###########################
    # Data for Machine Learning
    # Moving Average
    ma5 = Close.rolling(window=5).mean()
    ma20 = Close.rolling(window=20).mean()
    ma60 = Close.rolling(window=60).mean()
    ma120 = Close.rolling(window=120).mean()

    # Exponential Moving Average
    ema5 = Close.apply(lambda x: talib.func.EMA(x, 5), axis=0)
    ema20 = Close.apply(lambda x: talib.func.EMA(x, 20), axis=0)
    ema60 = Close.apply(lambda x: talib.func.EMA(x, 60), axis=0)
    ema120 = Close.apply(lambda x: talib.func.EMA(x, 120), axis=0)

    # Bollinger Bands
    mstd20 = Close.rolling(window=20).std()
    bb_up = ma20 + mstd20 * 2
    bb_dn = ma20 - mstd20 * 2
    # bb_up = Close.apply(lambda x: talib.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], axis=0)
    # bb_dn = Close.apply(lambda x: talib.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], axis=0)

    # RSI
    rsi14 = Close.apply(lambda x: talib.RSI(x, 14), axis=0)

    # MACD
    macd = Close.apply(lambda x: talib.MACD(x, 12, 26, 9)[0])
    # macd_sig = Close.apply(lambda x: talib.MACD(x, 12, 26, 9)[1])

    # Momentum(Log Returns for x period)
    mom20 = (np.log(Close)).diff(20)
    mom60 = (np.log(Close)).diff(60)
    mom120 = (np.log(Close)).diff(120)

    ###########################
    #     Set Time Period     #
    ###########################
    # Set Formation & Trading Period
    d = datetime.datetime(2011, 1, 1)
    date_index = Close.index[Close.index > d]
    year_list = date_index.year.unique()
    # Formation period: 12 months, Trading period: 6 months
    trading_period = relativedelta(months=6)
    formation_period = relativedelta(months=12)
    total_period = trading_period + formation_period

    formation_str_list = ([datetime.datetime(y, 1, 1) for y in year_list if
                           datetime.datetime(y, 1, 1) + total_period < date_index[-1]] +
                          [datetime.datetime(y, 7, 1) for y in year_list if
                           datetime.datetime(y, 7, 1) + total_period < date_index[-1]])
    formation_str_list = sorted(formation_str_list)
    trading_str_list = [x + formation_period for x in formation_str_list]

    formation_end_list = [x + formation_period + relativedelta(days=-1) for x in formation_str_list]
    trading_end_list = [x + trading_period + relativedelta(days=-1) for x in trading_str_list]

    ###########################
    #         Trading         #
    ###########################
    stat_list = []
    for fs, fe, ts, te in zip(formation_str_list, formation_end_list, trading_str_list, trading_end_list):
        # Copy the target period data
        df_price = Close.loc[fs:te, :].copy()
        df_volume = Volume.loc[fs:te, :].copy()
        df_ma5 = ma5.loc[fs:te, :].copy()
        df_ma20 = ma20.loc[fs:te, :].copy()
        df_ma60 = ma60.loc[fs:te, :].copy()
        df_ma120 = ma120.loc[fs:te, :].copy()
        df_ema5 = ema5.loc[fs:te, :].copy()
        df_ema20 = ema20.loc[fs:te, :].copy()
        df_ema60 = ema60.loc[fs:te, :].copy()
        df_ema120 = ema120.loc[fs:te, :].copy()
        df_bbu = bb_up.loc[fs:te, :].copy()
        df_bbd = bb_dn.loc[fs:te, :].copy()
        df_rsi14 = rsi14.loc[fs:te, :].copy()
        df_macd = macd.loc[fs:te, :].copy()
        df_mom20 = mom20.loc[fs:te, :].copy()
        df_mom60 = mom60.loc[fs:te, :].copy()
        df_mom120 = mom120.loc[fs:te, :].copy()

        # Drop NA Stocks
        df_price = df_price.dropna(axis=1)
        df_volume = df_volume.dropna(axis=1)
        df_ma5 = df_ma5.dropna(axis=1)
        df_ma20 = df_ma20.dropna(axis=1)
        df_ma60 = df_ma60.dropna(axis=1)
        df_ma120 = df_ma120.dropna(axis=1)
        df_ema5 = df_ema5.dropna(axis=1)
        df_ema20 = df_ema20.dropna(axis=1)
        df_ema60 = df_ema60.dropna(axis=1)
        df_ema120 = df_ema120.dropna(axis=1)
        df_bbu = df_bbu.dropna(axis=1)
        df_bbd = df_bbd.dropna(axis=1)
        df_rsi14 = df_rsi14.dropna(axis=1)
        df_macd = df_macd.dropna(axis=1)
        df_mom20 = df_mom20.dropna(axis=1)
        df_mom60 = df_mom60.dropna(axis=1)
        df_mom120 = df_mom120.dropna(axis=1)

        print('{} ~ {}'.format(fs, te))
        print('# of Stocks: {}'.format(len(df_price.columns)))

        formation_close, trading_close = df_price[fs:fe], df_price[ts:te]
        # Data for Machine Learning
        formation_volume, trading_volume = df_volume[fs:fe], df_volume[ts:te]
        formation_ma5, trading_ma5 = df_ma5[fs:fe], df_ma5[ts:te]
        formation_ma20, trading_ma20 = df_ma20[fs:fe], df_ma20[ts:te]
        formation_ma60, trading_ma60 = df_ma60[fs:fe], df_ma60[ts:te]
        formation_ma120, trading_ma120 = df_ma120[fs:fe], df_ma120[ts:te]

        formation_ema5, trading_ema5 = df_ema5[fs:fe], df_ema5[ts:te]
        formation_ema20, trading_ema20 = df_ema20[fs:fe], df_ema20[ts:te]
        formation_ema60, trading_ema60 = df_ema60[fs:fe], df_ema60[ts:te]
        formation_ema120, trading_ema120 = df_ema120[fs:fe], df_ema120[ts:te]

        formation_bbu, trading_bbu = df_bbu[fs:fe], df_bbu[ts:te]
        formation_bbd, trading_bbd = df_bbd[fs:fe], df_bbd[ts:te]
        formation_rsi14, trading_rsi14 = df_rsi14[fs:fe], df_rsi14[ts:te]
        formation_macd, trading_macd = df_macd[fs:fe], df_macd[ts:te]

        formation_mom20, trading_mom20 = df_mom20[fs:fe], df_mom20[ts:te]
        formation_mom60, trading_mom60 = df_mom60[fs:fe], df_mom60[ts:te]
        formation_mom120, trading_mom120 = df_mom120[fs:fe], df_mom120[ts:te]

        ml_list = [
            (formation_volume, trading_volume),
            # (formation_ma5, trading_ma5),
            # (formation_ma20, trading_ma20),
            # (formation_ma60, trading_ma60),
            # (formation_ma120, trading_ma120),
            # (formation_ema5, trading_ema5),
            (formation_ema20, trading_ema20),
            # (formation_ema60, trading_ema60),
            # (formation_ema120, trading_ema120),
            # (formation_bbu, trading_bbu),
            # (formation_bbd, trading_bbd),
            (formation_rsi14, trading_rsi14),
            (formation_macd, trading_macd),
            (formation_mom20, trading_mom20),
            # (formation_mom60, trading_mom60),
        ]

        # stat = pairs_trading(formation_close, trading_close, ml_list=ml_list)
        stat = pairs_trading(formation_close, trading_close)
        stat.print_statistics()
        stat_list.append(stat)
    # pd.to_pickle(stat_list, './pickle/stat_optics(3)_ma_xgb_log3.pkl')
    # pd.to_pickle(stat_list, './pickle/stat_optics(3)_lim5_2.pkl')
    # pd.to_pickle(stat_list, './pickle/kmeans(2^)_lim5_limp10_ret.pkl')
    # pd.to_pickle(stat_list, './pickle/kmeans(2^)_lim7_limp10_bugfix.pkl')
    pd.to_pickle(stat_list, './pickle/optics_dbscan(.5_2_4).pkl')
    # pd.to_pickle(stat_list, './pickle/optics_dbscan(.25_1)_lim3_limp10_bugfix.pkl')
    # pd.to_pickle(stat_list, './pickle/dbscan(.01_3)_lim7_limp10_ret.pkl')
    # pd.to_pickle(stat_list, './pickle/dbscan(.01_3)_lim8_limp10.pkl')


if __name__ == func_o:
    ###########################
    #       Data Loading      #
    ###########################
    root_path = 'pickle'
    ticker_pkl = 'kse_18_20_ticker.pkl'
    sector_pkl = 'kse_sector.pkl'
    # price_pkl = 'kse_18_20.pkl'
    # price_pkl = 'kse_11_20.pkl'
    price_pkl = 'kse_01_20_volume.pkl'
    ticker_pkl = os.path.join(root_path, ticker_pkl)
    sector_pkl = os.path.join(root_path, sector_pkl)
    price_pkl = os.path.join(root_path, price_pkl)

    ticker_series = pd.read_pickle(ticker_pkl)
    sector_df = pd.read_pickle(sector_pkl)
    df_ori = pd.read_pickle(price_pkl)

    Close = df_ori['수정주가(원)']
    Volume = df_ori['거래량(주)']
    Cap = df_ori['시가총액 (52주 평균)(백만원)']
    assert (Close.isnull().sum()).equals(Volume.isnull().sum())

    ###########################
    #   Data Pre-processing   #
    ###########################
    # Data for Machine Learning
    # Moving Average
    ma5 = Close.rolling(window=5).mean()
    ma20 = Close.rolling(window=20).mean()
    ma60 = Close.rolling(window=60).mean()
    ma120 = Close.rolling(window=120).mean()

    # Exponential Moving Average
    ema5 = Close.apply(lambda x: talib.func.EMA(x, 5), axis=0)
    ema20 = Close.apply(lambda x: talib.func.EMA(x, 20), axis=0)
    ema60 = Close.apply(lambda x: talib.func.EMA(x, 60), axis=0)
    ema120 = Close.apply(lambda x: talib.func.EMA(x, 120), axis=0)

    # Bollinger Bands
    mstd20 = Close.rolling(window=20).std()
    bb_up = ma20 + mstd20 * 2
    bb_dn = ma20 - mstd20 * 2
    # bb_up = Close.apply(lambda x: talib.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[0], axis=0)
    # bb_dn = Close.apply(lambda x: talib.BBANDS(x, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)[2], axis=0)

    # RSI
    rsi14 = Close.apply(lambda x: talib.RSI(x, 14), axis=0)

    # MACD
    macd = Close.apply(lambda x: talib.MACD(x, 12, 26, 9)[0])
    # macd_sig = Close.apply(lambda x: talib.MACD(x, 12, 26, 9)[1])

    # Momentum(Log Returns for x period)
    mom20 = (np.log(Close)).diff(20)
    mom60 = (np.log(Close)).diff(60)
    mom120 = (np.log(Close)).diff(120)

    # Set Formation & Trading Period
    """
    fs = datetime.datetime(2018, 7, 1)
    fe = datetime.datetime(2019, 6, 30)
    ts = datetime.datetime(2019, 7, 1)
    te = datetime.datetime(2019, 12, 31)

    """
    fs = datetime.datetime(2014, 1, 1)
    fe = datetime.datetime(2014, 12, 31)
    ts = datetime.datetime(2015, 1, 1)
    te = datetime.datetime(2015, 6, 30)

    # Copy the target period data
    df_price = Close.loc[fs:te, :].copy()
    df_volume = Volume.loc[fs:te, :].copy()
    df_ma5 = ma5.loc[fs:te, :].copy()
    df_ma20 = ma20.loc[fs:te, :].copy()
    df_ma60 = ma60.loc[fs:te, :].copy()
    df_ma120 = ma120.loc[fs:te, :].copy()
    df_ema5 = ema5.loc[fs:te, :].copy()
    df_ema20 = ema20.loc[fs:te, :].copy()
    df_ema60 = ema60.loc[fs:te, :].copy()
    df_ema120 = ema120.loc[fs:te, :].copy()
    df_bbu = bb_up.loc[fs:te, :].copy()
    df_bbd = bb_dn.loc[fs:te, :].copy()
    df_rsi14 = rsi14.loc[fs:te, :].copy()
    df_macd = macd.loc[fs:te, :].copy()
    df_mom20 = mom20.loc[fs:te, :].copy()
    df_mom60 = mom60.loc[fs:te, :].copy()
    df_mom120 = mom120.loc[fs:te, :].copy()

    # Drop NA Stocks
    df_price = df_price.dropna(axis=1)
    df_volume = df_volume.dropna(axis=1)
    df_ma5 = df_ma5.dropna(axis=1)
    df_ma20 = df_ma20.dropna(axis=1)
    df_ma60 = df_ma60.dropna(axis=1)
    df_ma120 = df_ma120.dropna(axis=1)
    df_ema5 = df_ema5.dropna(axis=1)
    df_ema20 = df_ema20.dropna(axis=1)
    df_ema60 = df_ema60.dropna(axis=1)
    df_ema120 = df_ema120.dropna(axis=1)
    df_bbu = df_bbu.dropna(axis=1)
    df_bbd = df_bbd.dropna(axis=1)
    df_rsi14 = df_rsi14.dropna(axis=1)
    df_macd = df_macd.dropna(axis=1)
    df_mom20 = df_mom20.dropna(axis=1)
    df_mom60 = df_mom60.dropna(axis=1)
    df_mom120 = df_mom120.dropna(axis=1)

    print('{} ~ {}'.format(fs, te))
    print('# of Stocks: {}'.format(len(df_price.columns)))

    formation_close, trading_close = df_price[fs:fe], df_price[ts:te]

    # Data for Machine Learning
    formation_volume, trading_volume= df_volume[fs:fe], df_volume[ts:te]
    formation_ma5, trading_ma5 = df_ma5[fs:fe], df_ma5[ts:te]
    formation_ma20, trading_ma20 = df_ma20[fs:fe], df_ma20[ts:te]
    formation_ma60, trading_ma60 = df_ma60[fs:fe], df_ma60[ts:te]
    formation_ma120, trading_ma120 = df_ma120[fs:fe], df_ma120[ts:te]

    formation_ema5, trading_ema5 = df_ema5[fs:fe], df_ema5[ts:te]
    formation_ema20, trading_ema20 = df_ema20[fs:fe], df_ema20[ts:te]
    formation_ema60, trading_ema60 = df_ema60[fs:fe], df_ema60[ts:te]
    formation_ema120, trading_ema120 = df_ema120[fs:fe], df_ema120[ts:te]

    formation_bbu, trading_bbu = df_bbu[fs:fe], df_bbu[ts:te]
    formation_bbd, trading_bbd = df_bbd[fs:fe], df_bbd[ts:te]
    formation_rsi14, trading_rsi14 = df_rsi14[fs:fe], df_rsi14[ts:te]
    formation_macd, trading_macd = df_macd[fs:fe], df_macd[ts:te]

    formation_mom20, trading_mom20 = df_mom20[fs:fe], df_mom20[ts:te]
    formation_mom60, trading_mom60 = df_mom60[fs:fe], df_mom60[ts:te]
    formation_mom120, trading_mom120 = df_mom120[fs:fe], df_mom120[ts:te]

    ml_list = [
        (formation_volume, trading_volume),
        # (formation_ma5, trading_ma5),
        # (formation_ma20, trading_ma20),
        # (formation_ma60, trading_ma60),
        # (formation_ma120, trading_ma120),
        # (formation_ema5, trading_ema5),
        (formation_ema20, trading_ema20),
        # (formation_ema60, trading_ema60),
        # (formation_ema120, trading_ema120),
        # (formation_bbu, trading_bbu),
        # (formation_bbd, trading_bbd),
        (formation_rsi14, trading_rsi14),
        (formation_macd, trading_macd),
        (formation_mom20, trading_mom20),
        # (formation_mom60, trading_mom60),
    ]

    # stat = pairs_trading(formation_close, trading_close, ml_list=ml_list)
    stat = pairs_trading(formation_close, trading_close)
    stat.print_statistics()
    pd.to_pickle(stat, './pickle/test.pkl')
