import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans, OPTICS, DBSCAN, cluster_optics_dbscan,\
    SpectralClustering, AffinityPropagation, MeanShift, Birch, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


def adf_coint_test(df1, df2):
    df2_temp = df2.copy()
    # df2_temp = sm.add_constant(df2_temp)
    results = sm.OLS(df1, df2_temp).fit()
    coint_pval = adfuller(results.resid)[1]
    # alpha = results.params[0]
    # beta = results.params[1]
    alpha = 0
    beta = results.params[0]
    r2 = results.rsquared

    return coint_pval, beta, alpha, r2


def find_cointegrated_pairs_adf(dataframe, critial_level=0.02):
    n = dataframe.shape[1]  # the length of dateframe
    pvalue_matrix = np.ones((n, n))  # initialize the matrix of p
    keys = dataframe.keys()  # get the column names
    pairs = []  # initilize the list for cointegration
    # validation = False
    validation = False
    for i in range(n):
        for j in range(i + 1, n):  # for j bigger than i
            # stock1 = np.log(dataframe[keys[i]])  # obtain the log price of two contract
            # stock2 = np.log(dataframe[keys[j]])
            stock1 = dataframe[keys[i]]  # obtain the price of two contract
            stock2 = dataframe[keys[j]]
            pval, beta, alpha, r2 = adf_coint_test(stock1, stock2)  # get conintegration
            pvalue_matrix[i, j] = pval
            # Validation Image
            if validation:
                stock1.plot()
                (beta*stock2 + alpha).plot()
                plt.title("R2: {}, P val: {}".format(r2, pval))
                plt.show()
            if pval < critial_level:  # if p-value less than the critical level
                pairs.append((keys[i], keys[j], pval, beta, alpha, r2))  # record the contract with that p-value

    return pvalue_matrix, pairs


def find_pairs(df_price):
    # Normalization
    df_return = df_price.pct_change()
    df_return = df_return.dropna(axis=0)
    """
    for col in df_return.columns:
        df_return[col] = (df_return[col] - df_return[col].mean()) / df_return[col].std()
    """
    scaler = StandardScaler()
    df_return_scaled = pd.DataFrame(scaler.fit_transform(df_return), columns=df_return.columns, index=df_return.index)

    # PCA on return space
    pca = PCA()
    pca.fit(df_return_scaled)
    return_pca = pca.transform(df_return_scaled)

    # Eigen portfolio returns
    df_eig = pd.DataFrame()
    eig_lim = 7
    for i in range(eig_lim):
        df_eig[i] = df_return_scaled.mul(pca.components_[i], axis=1).sum(axis=1)

    # Regression (X: Eigen portfolios(Risk Factor from PCA), Y: Individual Return)
    df_reg = df_eig.copy()
    df_reg = sm.add_constant(df_reg)

    # Factor Loading estimation(OLS) for each stocks
    df_params = pd.DataFrame()
    df_pval = pd.DataFrame()

    # Independent Variable of Regression should be df_return, not df_return_scaled
    for ticker in df_return.columns:
        results = sm.OLS(df_return.loc[:, ticker], df_reg).fit()
        df_params[ticker] = results.params
        df_pval[ticker] = results.pvalues

    df_params = df_params.T
    df_pval = df_pval.T

    # First Clustering: Grouping stocks which have same significant factors
    pval_thr = 0.05
    df_pval_bool = df_pval < pval_thr

    def classify_duplicate(df):
        df_c = df.copy()
        df_list = []
        df_non = pd.DataFrame()
        idx_list = df.index
        for idx in idx_list:
            if idx not in df_c.index:
                continue
            # print(idx)
            temp_df = pd.DataFrame()
            # temp_df[idx] = df_c.loc[idx,:]
            temp_series = df_c.loc[idx, :]
            df_c = df_c.drop(idx)
            cnt = 0
            for idx2 in df_c.index:
                if (temp_series == df_c.loc[idx2, :]).all():
                    # print(idx, idx2)
                    temp_df[idx2] = df_c.loc[idx2, :]
                    df_c = df_c.drop(idx2)
                    cnt += 1
            if cnt == 0:
                df_non[idx] = temp_series
            else:
                temp_df[idx] = temp_series
                df_list.append(temp_df.T)
        return df_list, df_non.T

    classified_list, df_nc = classify_duplicate(df_pval_bool)

    # Second Clustering: Grouping stocks which have similar factor loadings
    for d in classified_list:  # [:10]:
        target_tickers = d.index
        factor_sig = d.iloc[0, :]
        if not (factor_sig == True).any():
            continue
        # k = factor_sig.sum()
        exclude_mkt = ~factor_sig.index.isin([0])  # Exclude 1st Eig Port(Market Index)
        k = 2**(factor_sig[exclude_mkt].sum())

        sig_factor = list(factor_sig.index[factor_sig])
        assert(sig_factor != [])
        if 'const' in sig_factor:
            continue
        d['sig_factor'] = [sig_factor] * len(d)
        if len(d) < k*2 or k < 2 or len(d) < 5:
            continue

        clustering_algo = {
            'Kmeans': KMeans(n_clusters=k, init='k-means++', random_state=1),
            # 'OPTICS': OPTICS(min_samples=3, max_eps=0.1),
            # 'OPTICS': OPTICS(min_samples=5),#, max_eps=0.05),
            'OPTICS': OPTICS(min_samples=2),#, max_eps=0.05),
            'OPTICS_1': OPTICS(min_samples=k, max_eps=0.1),
            'OPTICS_2': OPTICS(min_samples=int(np.sqrt(k)), max_eps=0.1),
            'AC': AgglomerativeClustering(),
            'AP': AffinityPropagation(),
            'SC': SpectralClustering(),
            'Birch': Birch(),
            'MS': MeanShift(),
            'DBSCAN': DBSCAN(eps=0.01, min_samples=3)
        }

        def optics_dbscan(X):
            # validation = True
            validation = False
            clust = clustering_algo['OPTICS']
            clust.fit(X)
            optics_params = X.copy()
            optics_params['cluster'] = clust.labels_
            optics_params['order'] = clust.ordering_
            optics_params['rd'] = clust.reachability_
            optics_params['cd'] = clust.core_distances_

            rd_diff = optics_params['rd'].sort_values().diff()
            exclude_list = [np.nan, np.inf, -np.inf]
            rd_diff_mean = rd_diff[~rd_diff.isin(exclude_list)].mean()
            rd_diff_std = rd_diff[~rd_diff.isin(exclude_list)].std()
            eps_thr = rd_diff_mean # + rd_diff_std
            # eps_thr = rd_diff[~rd_diff.isin(exclude_list)].mean()
            rd_outlier = rd_diff > eps_thr
            sig_eps_num = int(len(rd_diff)*0.30)
            rd_outlier = rd_outlier[sig_eps_num:]
            cutoff1 = rd_outlier[rd_outlier == True].index[0]
            cutoff0 = rd_diff[:cutoff1].index[-2]
            eps = (optics_params.loc[cutoff0, 'rd'] + optics_params.loc[cutoff1, 'rd'])/2
            labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                            core_distances=clust.core_distances_,
                                            ordering=clust.ordering_, eps=eps)
            if validation:
                plt.figure(figsize=(10, 7))
                G = gridspec.GridSpec(1, 1)
                ax1 = plt.subplot(G[0, :])

                labels = clust.labels_[clust.ordering_]
                reachability = clust.reachability_
                space = np.arange(len(optics_params))
                colors = ['g.', 'r.', 'b.', 'y.', 'c.', 'o.', 'k.']
                for klass, color in zip(range(0, 5), colors):
                    Xk = space[labels == klass]
                    Rk = reachability[labels == klass]
                    ax1.plot(Xk, Rk, color, alpha=0.3)
                ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.7)
                ax1.set_ylabel('Reachability (epsilon distance)')
                ax1.set_title('Reachability Plot')
                plt.bar(optics_params.index, optics_params['rd'])
                plt.show()

                optics_params.sort_values(by='rd').rd.plot(kind='bar', alpha=.5)
                optics_params.sort_values(by='rd').rd.diff().plot(kind='bar', alpha=.5, color='orange')
                # params.sort_values(by='rd').cd.plot(kind='bar', color='orange', alpha=.5)
                plt.axhline(y=eps)
                plt.show()
            return labels

        kmeans = clustering_algo['Kmeans']
        kmeans.fit(df_params.loc[target_tickers, factor_sig == True])
        # gmm = GaussianMixture(n_components=k)
        # gmm.fit(df_params.loc[target_tickers, factor_sig == True])
        # clustering = clustering_algo['OPTICS']
        clustering = clustering_algo['DBSCAN']
        clustering.fit(df_params.loc[target_tickers, factor_sig == True])
        # d['cluster'] = kmeans.labels_
        # d['cluster'] = clustering.labels_
        d['cluster'] = optics_dbscan(df_params.loc[target_tickers, factor_sig == True])
        """
        if k == 2:
            d['cluster'] = kmeans.labels_
        else:
            d['cluster'] = optics_dbscan(df_params.loc[target_tickers, factor_sig == True])
        """
        # d['cluster'] = gmm.predict(df_params.loc[target_tickers, factor_sig == True])

        # Validation code with Image
        if k == -1:
            tc = factor_sig == True
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 'magenta', 'black', 'cyan', 'black', 'gray','pink']

            colors = [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

            # kclist = [colors[x] for x in kmeans.labels_]
            # oclist = [colors[x] for x in clustering.labels_]
            # gclist = [colors[x] for x in gmm.predict(df_params.loc[target_tickers, factor_sig == True])]
            plt.title('K-Means')
            plt.scatter(df_params.loc[target_tickers, tc].iloc[:, 1], df_params.loc[target_tickers, tc].iloc[:, 2],
                        c=(kmeans.labels_ + 1), cmap=plt.cm.tab20)
            plt.colorbar()
            plt.show()
            plt.title('OPTICS - {} labels'.format(len(set(clustering.labels_))))
            plt.scatter(df_params.loc[target_tickers, tc].iloc[:, 1], df_params.loc[target_tickers, tc].iloc[:, 2],
                        c=(d['cluster']), cmap=plt.cm.tab20)
            plt.colorbar()
            plt.show()

    df_pairs_total = pd.DataFrame(columns=['s1', 's2', 'pval', 'beta', 'alpha', 'sig_factor', 'cluster'])
    # Final Round for Finding Pairs: Cointegration
    for i, c1 in enumerate(classified_list):
        if 'sig_factor' not in c1.columns:
            # There is no significant Risk Factor
            continue
        if 'cluster' not in c1.columns:
            # Not Clustered stocks in first clustering step
            continue
        if len(c1) < 3:
            continue
        sig_factor = c1['sig_factor'][0]
        df_pairs_semi = pd.DataFrame(columns=['s1', 's2', 'pval', 'beta', 'alpha', 'sig_factor', 'cluster'])
        print("Finding Pairs from {} Cluster({} stocks in the cluster)..".format(i, len(c1)))

        for c2 in c1['cluster'].unique():
            if c2 == -1:
                # print('Noise')
                continue
            print("\tFinding Pairs from {} Small Cluster".format(c2))
            stocks = c1.index[c1['cluster'] == c2]
            # Cointegration test based on Prices
            # _, pairs_p = find_cointegrated_pairs_adf(df_price.loc[:, stocks])

            # Based Cointegration test based on Returns
            _, pairs_r = find_cointegrated_pairs_adf((df_return.loc[:, stocks] + 1).cumprod())

            pairs = pairs_r
            df_pairs = pd.DataFrame(pairs, columns=['s1', 's2', 'pval', 'beta', 'alpha', 'r2'])
            df_pairs = df_pairs.sort_values(by='pval').reset_index(drop=True)
            df_pairs['cluster'] = str(i) + '_' + str(c2)
            df_pairs['sig_factor'] = [sig_factor] * len(df_pairs)
            df_pairs_semi = df_pairs_semi.append(df_pairs, ignore_index=True)

        if len(df_pairs_semi) == 0:  # If c1 has only one cluster and it was -1(Noise),
            continue

        # Select Top n Pairs from each Clusters
        r2_high_thr = 1
        r2_low_thr = 0.75
        # r2_thr = 0.9
        df_pairs_semi = df_pairs_semi.loc[df_pairs_semi['r2'] > r2_low_thr, :]
        df_pairs_semi = df_pairs_semi.loc[df_pairs_semi['r2'] < r2_high_thr, :]
        top_n = 10
        if len(df_pairs_semi) > top_n:
            df_pairs_semi = df_pairs_semi[:top_n]
        df_pairs_total = df_pairs_total.append(df_pairs_semi, ignore_index=True)

        # Final Filtering
        # df_pairs_total = df_pairs_total.sort_values(by='pval').reset_index(drop=True)
        # if len(df_pairs_total) > 30:
            # df_pairs_total = df_pairs_total[:30]

    return df_pairs_total, pca.components_[:eig_lim]


def draw_pairs(df_pairs, df_p):
    # font_path = 'C:/Windows/Fonts/H2GTRM.TTF'
    # fontprop = FontProperties(fname=font_path, size=15)
    for i in df_pairs.index:
        s1_code = df_p[df_pairs.loc[i, 's1']].name
        s2_code = df_p[df_pairs.loc[i, 's2']].name
        s1_name = s1_code  # ticker_series[s1_code]
        s2_name = s2_code  # ticker_series[s2_code]

        logp1 = (df_p[df_pairs.loc[i, 's1']])
        logp2 = (df_p[df_pairs.loc[i, 's2']])
        a = df_pairs.loc[i, 'alpha']
        b = df_pairs.loc[i, 'beta']

        pd.Series(logp1).plot(label=s1_name)
        pd.Series(logp2 * b + a).plot(label=s2_name)
        print('{}th'.format(i))
        print(df_pairs.loc[i, 'cluster'])
        print(df_pairs.loc[i, 'pval'])
        print('R2: ' + str(df_pairs.loc[i, 'r2']))

        plt.legend()
        plt.show()
        plt.close()
