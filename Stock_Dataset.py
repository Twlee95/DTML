from pandas_datareader import data as pdr
import yfinance as yfin

class StockDataset:
    def __init__(self, symbol_list, x_frames, y_frames,start, end):
        self.symbol_list = symbol_list
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.start = start
        self.end = end
        self.data_list_ = self.data_list(self.symbol_list, self.start, self.end)

    def data_list(self, symbol_list, start, end):
        data_list = []
        for symbol in symbol_list:
            yfin.pdr_override()
            data = pdr.get_data_yahoo(symbol, start=start, end=end)
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list_[0])

    def __getitem__(self, idx):
        X_list=[]
        y_list=[]
        for data in self.data_list_:
            X = data.iloc[idx:idx + self.x_frames].to_numpy()
            y = data.iloc[idx + self.x_frames:idx + self.x_frames+self.y_frames].to_numpy()
            X_list.append(X)
            y_list.append(y)
        return X_list, y_list  ## 데이터별 리스트

if __name__=="__main__":
    data_symbol_list = ['^IXIC', 'AAPL', 'AMZN','MSFT','TSLA','GOOG','FB','NVDA','AMD']
    trainset = StockDataset(data_symbol_list, 10, 1,"2012-01-01","2016-12-31")





