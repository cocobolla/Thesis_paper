import numpy as np


def sharpe_ratio(ret_series, rf_rate=0.02):
    yr1 = 252
    # daily_rf = rf_rate / yr1
    daily_rf = (1+rf_rate)**(1.0/yr1) - 1
    s = (ret_series.mean() - daily_rf) / ret_series.std()
    annual_s = s * np.sqrt(yr1)
    return annual_s


class Statistics:
    def __init__(self, ret_df):
        self.ret_df = ret_df
        ret_list = ret_df.cumprod().iloc[-1, :]
        self.ret_mean = ret_list.mean()
        self.ret_std = ret_list.std()
        self.ret_skew = ret_list.skew()
        self.ret_kur = ret_list.kurtosis()
        self.ret_min = ret_list.min()
        self.ret_max = ret_list.max()
        self.roo = (ret_list > 1).sum()
        self.ruo = (ret_list < 1).sum()
        self.reo = (ret_list == 1).sum()
        rf = 0.02
        daily_ret = (ret_df - 1).sum(axis=1)
        self.sharpe = sharpe_ratio(daily_ret, rf)

    def print_statistics(self):
        print("Date: {} ~ {}".format(self.ret_df.index[0], self.ret_df.index[-1]))
        print('# of Date: {}'.format(len(self.ret_df)))
        print('# of Pairs: {}'.format(len(self.ret_df.columns)))
        print('Mean: {:.4f}'.format(self.ret_mean))
        print('Standard Deviation: {:.4f}'.format(self.ret_std))
        print('Skewness: {:.4f}'.format(self.ret_skew))
        print('Kurtosis: {:.4f}'.format(self.ret_kur))
        print('Min: {:.4f}'.format(self.ret_min))
        print('Max: {:.4f}'.format(self.ret_max))
        print('Observation with Excess Return > 1: {} ({:.2f}%)'.format(self.roo, self.roo/len(self.ret_df.columns)))
        print('Observation with Excess Return < 1: {} ({:.2f}%)'.format(self.ruo, self.ruo/len(self.ret_df.columns)))
        print('Observation with Excess Return = 1: {} ({:.2f}%)'.format(self.reo, self.reo/len(self.ret_df.columns)))
        print('Sharpe Ratio: {:.4f}'.format(self.sharpe))
