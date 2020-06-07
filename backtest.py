import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import reciprocal, uniform


import talib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
sd_weight = {
    'Close': 0.5,
    'Open': 1.5,
    'Loss Cut': 4
}
"""

sd_weight = {
    'Close': 0.5,
    'Open': 2,
    'Loss Cut': 4
}


def set_status(spread, mu, std):
    if spread > mu + std * sd_weight['Loss Cut']:
        return 1
    elif spread > mu + std * sd_weight['Open']:
        return 2
    elif spread > mu + std * sd_weight['Close']:
        return 3
    elif mu - std * sd_weight['Close'] < spread < mu + std * sd_weight['Close']:
        return 4
    elif spread > mu - std * sd_weight['Open']:
        return 5
    elif spread > mu - std * sd_weight['Loss Cut']:
        return 6
    else:
        return 7


def get_pair_returns(pair_info, formation_close, trading_close):
    position_count = 0
    pair1 = pair_info['s1']
    pair2 = pair_info['s2']
    beta = pair_info['beta']
    alpha = pair_info['alpha']

    # Get Data
    formation_price = formation_close.loc[:, [pair1, pair2]]
    trading_price = trading_close.loc[:, [pair1, pair2]]

    # If we trade with dollar neutral, It seems like we notice the return's spread
    formation_return = (formation_close.loc[:, [pair1, pair2]].pct_change() + 1).cumprod()
    trading_return = (trading_close.loc[:, [pair1, pair2]].pct_change() + 1).cumprod()

    formation = formation_return
    trading = trading_return

    # Formation Period Spread
    fspread = formation[pair1] - (formation[pair2] * beta + alpha)
    fspread_mu = np.mean(fspread)
    fspread_sd = np.std(fspread)

    # Trading Period Spread
    tspread = trading[pair1] - (trading[pair2] * beta + alpha)
    trading['spread'] = (tspread - fspread_mu) / fspread_sd  # Normalized(with formation statistics) spread
    trading['status'] = trading.apply(lambda x: set_status(x['spread'], 0, 1), axis=1)

    # trading['spread_ret'] = trading['stock'].pct_change().fillna(0)
    trading['pair1_ret'] = trading[pair1].pct_change().fillna(0)
    trading['pair2_ret'] = (trading[pair2] * beta + alpha).pct_change().fillna(0)
    trading['pair1_p'] = trading[pair1]
    trading['pair2_p'] = trading[pair2] * beta + alpha

    trading['portfolio_ret'] = 0.5 * (trading['pair1_ret'] - trading['pair2_ret'])
    trading['position'] = None
    trading['ml_position'] = None
    trading['dollar_neutral'] = 1
    # trading['pred'] = y_pred

    weighted_asset = 1.0
    equal_asset = 1.0

    # Set Position
    for i in range(1, len(trading)):
        prev_status = trading['status'][i - 1]
        now_status = trading['status'][i]
        prev_position = trading['position'][i - 1]

        if now_status == 1:
            trading['position'][i] = None
            break
        if now_status == 2:
            # trading['position'][i] = 'short' if trading['pred'][i] == 0 else None
            if (prev_status == 3) or (prev_position == 'short'):
                trading['position'][i] = 'short'
        if now_status == 3:
            if prev_position == 'short':
                trading['position'][i] = 'short'
        if now_status == 4:
            trading['position'][i] = None
        if now_status == 5:
            if prev_position == 'long':
                trading['position'][i] = 'long'
        if now_status == 6:
            # trading['a'] = (prev_status == 5) || (prev_position=='long')
            if (prev_status == 5) or (prev_position == 'long'):
                trading['position'][i] = 'long'  # if trading['pred'][i] == 1 else None
        if now_status == 7:
            trading['position'][i] = None  # Original project was short
            break

    trading.loc[(trading['position'] == 'short') | (trading['position'].shift() == 'short'), 'position'] = 'short'
    trading.loc[(trading['position'] == 'long') | (trading['position'].shift() == 'long'), 'position'] = 'long'

    long_index = (trading['position'] == 'long')
    short_index = (trading['position'] == 'short')

    # Dollar Neutral Returns
    trading['dollar_neutral'][long_index] += trading['portfolio_ret'][long_index]
    trading['dollar_neutral'][short_index] -= trading['portfolio_ret'][short_index]
    # trading['dollar_neutral'] = trading['dollar_neutral'].cumprod()

    # Quantity Neutral Returns
    trading['quantity_neutral'] = 1

    # Set open position number
    open_idx = long_index | short_index
    trading['count'] = 0
    trading.loc[(open_idx - open_idx.shift()) == 1, 'count'] = ((open_idx - open_idx.shift()) == 1).cumsum()
    trading.loc[(open_idx == True) & (trading['count'] == 0), 'count'] = np.nan
    trading['count'] = trading['count'].fillna(method='ffill')
    trading['pv'] = 0

    for c in trading['count'].unique()[1:]:
        temp_trading = trading.loc[trading['count'] == c, :]
        # Initial value is margin of each pair's investment
        margin = 0.5
        initial_pv = (temp_trading['pair1_p'][0] + temp_trading['pair2_p'][0]) * margin
        trading.loc[temp_trading.index[0], 'pv'] = 0  #
        for t in range(1, len(temp_trading.index)):
            now_index = temp_trading.index[t]
            prev_index = temp_trading.index[t - 1]
            pv = (temp_trading.loc[now_index, 'pair1_p'] - temp_trading.loc[prev_index, 'pair1_p'] -
                  (temp_trading.loc[now_index, 'pair2_p'] - temp_trading.loc[prev_index, 'pair2_p']))
            if temp_trading['position'][0] == 'long':
                trading.loc[now_index, 'pv'] = pv
            elif temp_trading['position'][0] == 'short':
                trading.loc[now_index, 'pv'] = -pv
        trading.loc[temp_trading.index, 'quantity_neutral'] = trading.loc[temp_trading.index, 'pv'].cumsum()
        trading.loc[temp_trading.index, 'quantity_neutral'] += initial_pv
        trading.loc[temp_trading.index, 'quantity_neutral'] = trading.loc[
                                                                  temp_trading.index, 'quantity_neutral'].pct_change() + 1
        trading.loc[temp_trading.index[0], 'quantity_neutral'] = 1
    # trading['quantity_neutral'] = trading['quantity_neutral'].cumprod()

    # ret_list.append(trading['returns'])

    for j in range(1, len(trading)):
        if trading['position'][j - 1] is None and trading['position'][j] is not None:
            position_count += 1
    return trading, position_count


#################################
#   Func for Machine Learning   #
#################################
def get_ml_features(*args):
    feature_list = []
    for d in args:
        feature_list.append(d.values)
    X = np.hstack(feature_list)
    return X


def get_ml_label(y_series):
    alpha = 0.4
    fit_smoothing = SimpleExpSmoothing(y_series).fit(smoothing_level=alpha, optimized=False)
    # y_smoothing = fit_smoothing.fittedvalues
    # y_smoothing = y_series
    y_smoothing = y_series.rolling(window=10, center=True).mean()
    y_label = y_smoothing.diff().shift(-1)
    y_label[y_label.isnull() == False] = (y_label[y_label.isnull() == False] > 0) * 1
    # y_label = 1*(y_smoothing.diff() > 0).shift(-1)# .fillna(0)
    """
    y_series.plot(label='Original')
    y_smoothing.plot(label='Smoothing')
    plt.legend()
    plt.show()
    """
    return y_label


def get_pair_returns_ml(pair_info, formation_close, trading_close, ml_list, eig_list):
    position_count = 0
    pair1 = pair_info['s1']
    pair2 = pair_info['s2']
    beta = pair_info['beta']
    alpha = pair_info['alpha']
    sig_factor = pair_info['sig_factor']

    # Get Data
    formation = formation_close.loc[:, [pair1, pair2]]
    trading = trading_close.loc[:, [pair1, pair2]]

    # If we trade with dollar neutral, It seems like we notice the return's spread
    formation_return = (formation_close.loc[:, [pair1, pair2]].pct_change() + 1).cumprod()
    trading_return = (trading_close.loc[:, [pair1, pair2]].pct_change() + 1).cumprod()

    formation = formation_return
    trading = trading_return

    # Formation Period Spread
    fspread = formation[pair1] - (formation[pair2] * beta + alpha)
    fspread_log = np.log(formation[pair1]) - np.log(formation[pair2] * beta + alpha)
    fspread_mu = np.mean(fspread)
    fspread_sd = np.std(fspread)

    # Trading Period Spread
    tspread = trading[pair1] - (trading[pair2] * beta + alpha)
    tspread_log = np.log(trading[pair1]) - np.log(trading[pair2] * beta + alpha)
    trading['spread'] = (tspread - fspread_mu) / fspread_sd  # Normalized(with formation statistics) spread
    trading['status'] = trading.apply(lambda x: set_status(x['spread'], 0, 1), axis=1)

    # trading['spread_ret'] = trading['stock'].pct_change().fillna(0)
    trading['pair1_ret'] = trading[pair1].pct_change().fillna(0)
    trading['pair2_ret'] = (trading[pair2] * beta + alpha).pct_change().fillna(0)
    trading['pair1_p'] = trading[pair1]
    trading['pair2_p'] = trading[pair2] * beta + alpha

    trading['portfolio_ret'] = 0.5 * (trading['pair1_ret'] - trading['pair2_ret'])
    trading['position'] = None
    trading['ml_position'] = None
    trading['dollar_neutral'] = 1

    #########################
    #   ML Pre-processing   #
    #########################
    # Get ML Data
    formation_ml_list = [x[0] for x in ml_list]
    formation_ml_list = [x.loc[:, [pair1, pair2]] for x in formation_ml_list]
    trading_ml_list = [x[1] for x in ml_list]
    trading_ml_list = [x.loc[:, [pair1, pair2]] for x in trading_ml_list]

    formation_sig_eigport = eig_list[0].loc[:, sig_factor]
    formation_ml_list += [formation_sig_eigport]
    trading_sig_eigport = eig_list[1].loc[:, sig_factor]
    trading_ml_list += [trading_sig_eigport]

    # Training set
    X_train = get_ml_features(formation_close.loc[:, [pair1, pair2]], *formation_ml_list)
    y_train = get_ml_label(fspread)
    # y_train = get_ml_label(fspread_log)

    # Filtering na index in y
    not_na_index = y_train.isnull() == False
    target_loc = [y_train.index.get_loc(x) for x in y_train.index[not_na_index]]
    X_train = X_train[target_loc, :]
    y_train = y_train[not_na_index]

    # Normalization
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    # Test set
    X_test = get_ml_features(trading_close.loc[:, [pair1, pair2]], *trading_ml_list)
    y_test = get_ml_label(tspread)
    # y_test = get_ml_label(tspread_log)
    # y_test = get_ml_label((tspread-fspread_mu)/fspread_sd)
    X_test = scaler.transform(X_test)

    na_col = np.isnan(X_train).any(axis=0)
    X_train = X_train[:, ~na_col]
    X_test = X_test[:, ~na_col]

    # Model
    def logistic_training(X, y):
        y = y.astype('int')
        clf = LogisticRegression(solver='sag')
        clf.fit(X, y)
        return clf

    def svm_training(X, y, tuning=True):
        y = y.astype('int')
        param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
        # clf = svm.SVC(C=1, gamma=0.1, decision_function_shape='ovo', kernel='rbf', random_state=0)
        clf = svm.SVC(random_state=0, C=1, gamma=0.1, decision_function_shape='ovo', kernel='rbf')
        if tuning:
            rnd_search_cv = RandomizedSearchCV(clf, param_distributions, n_iter=100, verbose=0, cv=3, random_state=0)
            rnd_search_cv.fit(X, y)
            print("Best Params: {}".format(rnd_search_cv.best_params_))
            print("Best Score: {:.2f}".format(rnd_search_cv.best_score_))
            clf = rnd_search_cv.best_estimator_
        clf.fit(X, y)
        return clf

    def rf_training(X, y, tuning=False):
        y = y.astype('int')
        params = {
            'n_estimators': [50, 100, 300, 500],
            'min_samples_leaf': [4],
            # 'min_samples_split': [2, 5, 10],
            'min_samples_split': [2],
            'max_depth': [10, 30, 70]
        }
        # clf = RandomForestClassifier(n_estimators=30, max_features='auto', n_jobs=12, random_state=0)
        clf = RandomForestClassifier(n_estimators=100, max_features='auto', n_jobs=12, random_state=0,
                                     min_samples_leaf=4, min_samples_split=2, max_depth=30)
        if tuning:
            rnd_search_cv = RandomizedSearchCV(clf, params, n_iter=50, verbose=0, cv=3, random_state=0)
            # rnd_search_cv = GridSearchCV(clf, params, verbose=0, cv=3)
            rnd_search_cv.fit(X, y)
            print("Best Params: {}".format(rnd_search_cv.best_params_))
            print("Best Score: {:.2f}".format(rnd_search_cv.best_score_))
            clf = rnd_search_cv.best_estimator_
        clf.fit(X, y)
        return clf

    def xg_training(X, y, tuning=False):
        y = y.astype('int')
        params = {
            'n_estimators': [100, 300, 500],
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 2, 3, 5],
            # 'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 6, 9]
        }
        # clf = XGBClassifier(tree_method='gpu_hist', subsample=0.8, random_state=0)
        clf = XGBClassifier(tree_method='gpu_hist', subsample=0.8, random_state=0,
                            n_estimators=150, gamma=1.5, colsample_bytree=0.8, max_depth=8, min_child_weight=5)
        if tuning:
            rnd_search_cv = RandomizedSearchCV(clf, params, n_iter=50, verbose=0, cv=3, random_state=0)
            # rnd_search_cv = GridSearchCV(clf, params, verbose=0, cv=3)
            rnd_search_cv.fit(X, y)
            print("Best Params: {}".format(rnd_search_cv.best_params_))
            print("Best Score: {:.2f}".format(rnd_search_cv.best_score_))
            clf = rnd_search_cv.best_estimator_

        clf.fit(X, y)
        return clf

    model = xg_training(X_train, y_train)
    # rf = rf_training(X_train, y_train)
    # imp_values = pd.Series(rf.feature_importances_)
    # sns.barplot(x=imp_values.index, y=imp_values)
    # plt.show()

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5) * 1
    # y_pred = model.predict_proba(X_test)[:, 1]

    trading['pred'] = y_pred
    trading['test'] = y_test

    #########################
    # Set Trading Position  #
    #########################
    waiting_short = False
    waiting_long = False
    for i in range(1, len(trading)):
        prev_status = trading['status'][i - 1]
        now_status = trading['status'][i]
        prev_position = trading['position'][i - 1]

        if now_status == 1:
            trading['position'][i] = None
            break
        if now_status == 2:
            # trading['position'][i] = 'short' if trading['pred'][i] == 0 else None
            if (prev_status == 3) or (prev_status == 2 and waiting_short == True):
                if trading['pred'][i] == 0:
                # if trading['pred'][i] < 0.35:
                    trading['position'][i] = 'short'
                    waiting_short = False
                else:
                    waiting_short = True
            if prev_position == 'short':
                trading['position'][i] = 'short'
        if now_status == 3:
            if prev_position == 'short':
                trading['position'][i] = 'short'
            else:
                if waiting_short:
                    waiting_short = False
        if now_status == 4:
            trading['position'][i] = None
        if now_status == 5:
            if prev_position == 'long':
                trading['position'][i] = 'long'
            else:
                if waiting_long:
                    waiting_long = False
        if now_status == 6:
            if (prev_status == 5) or (prev_status == 6 and waiting_long == True):
                if trading['pred'][i] == 1:
                # if trading['pred'][i] > 0.65:
                    trading['position'][i] = 'long'  # if trading['pred'][i] == 1 else None
                    waiting_long = False
                else:
                    waiting_long = True
            if prev_position == 'long':
                trading['position'][i] = 'long'
        if now_status == 7:
            trading['position'][i] = None  # Original project was short
            break

    trading.loc[(trading['position'] == 'short') | (trading['position'].shift() == 'short'), 'position'] = 'short'
    trading.loc[(trading['position'] == 'long') | (trading['position'].shift() == 'long'), 'position'] = 'long'

    long_index = (trading['position'] == 'long')
    short_index = (trading['position'] == 'short')

    # Dollar Neutral Returns
    trading['dollar_neutral'][long_index] += trading['portfolio_ret'][long_index]
    trading['dollar_neutral'][short_index] -= trading['portfolio_ret'][short_index]
    # trading['dollar_neutral'] = trading['dollar_neutral'].cumprod()

    # Quantity Neutral Returns
    trading['quantity_neutral'] = 1

    # Set open position number
    open_idx = long_index | short_index
    trading['count'] = 0
    trading.loc[(open_idx - open_idx.shift()) == 1, 'count'] = ((open_idx - open_idx.shift()) == 1).cumsum()
    trading.loc[(open_idx == True) & (trading['count'] == 0), 'count'] = np.nan
    trading['count'] = trading['count'].fillna(method='ffill')
    trading['pv'] = 0

    for c in trading['count'].unique()[1:]:
        temp_trading = trading.loc[trading['count'] == c, :]
        # Initial value is margin of each pair's investment
        margin = 0.5
        initial_pv = (temp_trading['pair1_p'][0] + temp_trading['pair2_p'][0]) * margin
        trading.loc[temp_trading.index[0], 'pv'] = 0  #
        for t in range(1, len(temp_trading.index)):
            now_index = temp_trading.index[t]
            prev_index = temp_trading.index[t - 1]
            pv = (temp_trading.loc[now_index, 'pair1_p'] - temp_trading.loc[prev_index, 'pair1_p'] -
                  (temp_trading.loc[now_index, 'pair2_p'] - temp_trading.loc[prev_index, 'pair2_p']))
            if temp_trading['position'][0] == 'long':
                trading.loc[now_index, 'pv'] = pv
            elif temp_trading['position'][0] == 'short':
                trading.loc[now_index, 'pv'] = -pv
        trading.loc[temp_trading.index, 'quantity_neutral'] = trading.loc[temp_trading.index, 'pv'].cumsum()
        trading.loc[temp_trading.index, 'quantity_neutral'] += initial_pv
        trading.loc[temp_trading.index, 'quantity_neutral'] = trading.loc[
                                                                  temp_trading.index, 'quantity_neutral'].pct_change() + 1
        trading.loc[temp_trading.index[0], 'quantity_neutral'] = 1
    # trading['quantity_neutral'] = trading['quantity_neutral'].cumprod()

    # ret_list.append(trading['returns'])

    for j in range(1, len(trading)):
        if trading['position'][j - 1] is None and trading['position'][j] is not None:
            position_count += 1
    return trading, position_count
