import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
import pandas as pd


class DatasetBuilder:
    def build_dataset_simple(self, prediction_horizon, interval, data_path='../data/input_data.csv', train_pct=.8):
        data = pd.read_csv(f'input_data/{interval}.csv', index_col=0)
#         data = download_data(interval)
        #         data = pd.read_csv(data_path, index_col=0).iloc[::-1].reset_index(drop=True)
        # X = torch.tensor(data.values.astype(np.float32)).reshape(data.shape[1], data.shape[0])
        y = torch.tensor(data['close'].values.astype(np.float32))
        X = torch.tensor(data['close'].values.astype(np.float32))

        self.y_mean = y.mean()
        self.y_std = y.std()

        X = np.log(X)

        y = y[prediction_horizon:]
        y = y[np.newaxis, :, np.newaxis]
        X = X[:-prediction_horizon]
        X = X[np.newaxis, :, np.newaxis]

        X, y = torch.tensor(X), torch.tensor(y)

        # data split with standard scaling
        train_len = int(X.shape[1] * train_pct)
        X = (X - X.mean()) / X.std()
        X_train = X[:, :train_len, :]
        # X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = X[:, train_len:, :]
        # X_test = (X_test - X_train.mean()) / X_train.std()
        y = (y - y.mean()) / y.std()

        y_train = y[:, :train_len, :]
        y_test = y[:, train_len:, :]

        return X_train, y_train, X_test, y_test

    def unscale_y(self, y):
        return (y * self.y_std) + self.y_mean


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.inputs, self.outputs = X, y
        self.n_samples = X.shape[0]

    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.inputs[idx], self.outputs[idx]


def download_data(interval):
    API_KEY = 'CMSZIWYWAHTR01BR'
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    if 'min' in interval:
        data, meta_data = ts.get_intraday(symbol='TSLA',interval=interval, outputsize='full')
    else:
        data, meta_data = ts.get_daily(symbol='TSLA', outputsize='full')
      
    data.rename(columns={'1. open': 'open', '2. high':'high', '3. low':'low', '4. close':'close', '5. volume':'volume'}, inplace=True)
    df = data.iloc[::-1].reset_index(drop=True)
    df.to_csv(f'input_data/{interval}.csv')
    return df