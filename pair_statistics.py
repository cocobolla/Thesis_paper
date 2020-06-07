import numpy as np
import pandas as pd


def sharpe_ratio(ret_series, rf_rate=0.02):
    yr1 = 252
    daily_rf = rf_rate / yr1
    # daily_rf = (1+rf_rate)**(1.0/yr1) - 1
    s = (ret_series.mean() - daily_rf) / ret_series.std()
    annual_s = s * np.sqrt(yr1)
    return annual_s


class Statistics:
    def __init__(self, res_dict):
        self.res_dict = res_dict
        self.ret_df = pd.DataFrame()
        self.count_s = pd.Series()
        self.acc_s = pd.Series()
        for k, v in res_dict.items():
            self.ret_df[k] = v['quantity_neutral']
            ret_cnt = v.loc[v['count'] != 0, 'count']
            # Trading Count
            if len(ret_cnt) == 0:
                self.count_s.loc[k] = 0
            else:
                self.count_s.loc[k] = ret_cnt[-1]
            # ML Performance
            if 'pred' in v.columns:
                na_index = v['test'].isnull() == True
                acc = (v['pred'][~na_index] == v['test'][~na_index]).mean()
                self.acc_s.loc[k] = acc

        self.ret_list = self.ret_df.cumprod().iloc[-1, :]
        self.ret_mean = self.ret_list.mean()
        self.ret_std = self.ret_list.std()
        self.ret_skew = self.ret_list.skew()
        self.ret_kur = self.ret_list.kurtosis()
        self.ret_min = self.ret_list.min()
        self.ret_max = self.ret_list.max()
        self.roo = (self.ret_list > 1).sum()
        self.ruo = (self.ret_list < 1).sum()
        self.reo = (self.ret_list == 1).sum()
        self.pairs_num = len(self.ret_df.columns)
        self.date_num = len(self.ret_df)
        rf = 0.02
        self.daily_ret_sum = (self.ret_df - 1).sum(axis=1)
        self.daily_ret_mean = (self.ret_df - 1).mean(axis=1)
        self.sharpe = sharpe_ratio(self.daily_ret_mean, rf)
        # self.sharpe = sharpe_ratio(self.ret_df.cumprod().mean(axis=1).pct_change().dropna(), rf)

    def print_statistics(self):
        print("Date: {} ~ {}".format(self.ret_df.index[0], self.ret_df.index[-1]))
        print('# of Date: {}'.format(len(self.ret_df)))
        print('# of Pairs: {}'.format(self.pairs_num))
        print('# of Opening Position: {}'.format(self.count_s.sum()))
        # print('Mean of Opening Position: {}'.format(self.count_s.mean()))
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

        if len(self.acc_s) != 0:
            print('ML Accuracy: {:.4f}'.format(self.acc_s.mean()))
