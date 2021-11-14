from pandas_datareader import data as pdr
import yfinance as yfin
import pandas as pd
import numpy as np

class make_feature:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.data_load = self.data_(self.symbol, self.start, self.end)

    def data_(self, symbol, start, end):
        yfin.pdr_override()
        data = pdr.get_data_yahoo(symbol, start=start, end=end)
        return data

    def make_lt_feature(self, data):
        lt_feature_list = []
        time_terms = [5,10,15,20,25,30]
        for time_term in time_terms:

            data_len = len(data.loc[:, "Adj Close"])
            hidden = [sum(data.loc[:, "Adj Close"][data_len - i - time_term:data_len - i]) for i in range(data_len)]

            lis_ = []
            for i in range(len(hidden)):
                length = len(hidden)
                a = hidden[length - i - 1]
                lis_.append(a)
            len(lis_)

            zdk = np.array(lis_) / time_term * np.array(data.loc[:, "Adj Close"]) - 1
            lt_feature_list.append(zdk)
        return np.array(lt_feature_list).T

    def tgt_make(self, data):
        tgt = np.where(data.loc[:, "Adj Close"] >= data.loc[:, "Adj Close"].shift(periods=1, axis=0), 1, 0)
        return tgt

    def all_feature(self):

        data = self.data_load
        lt_feature = self.make_lt_feature(data)

        z_open = (data.loc[:, "Open"] / data.loc[:, "Close"] - 1).to_numpy()
        z_high = (data.loc[:, "High"] / data.loc[:, "Close"] - 1).to_numpy()
        z_low = (data.loc[:, "Low"] / data.loc[:, "Close"] - 1).to_numpy()
        z_close = (data.loc[:, "Close"] / data.loc[:, "Close"].shift(periods=1, axis=0) - 1).to_numpy()
        z_adjclose = (data.loc[:, "Adj Close"] / data.loc[:, "Adj Close"].shift(periods=1, axis=0) - 1).to_numpy()


        tgt = self.tgt_make(data)

        df = np.column_stack((z_open, z_high, z_low, z_close, z_adjclose,lt_feature,tgt))

        return pd.DataFrame(df[29:])



if __name__=="__main__":
    mf = make_feature('^IXIC',"2012-01-01","2016-12-31")
    df = mf.all_feature()
    print(df)


    mf = make_feature('^IXIC',"2012-01-01","2016-12-31")
    df = mf.all_feature()
    print(df)

