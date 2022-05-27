# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/docs/stable/data.html
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

#from utils.tools import StandardScaler
# from utils.timefeatures import time_features
from model.timefeatures import time_features

import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path, features, targets, first_diff, flag='train',
                 size=None, scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.targets = targets
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.first_diff=first_diff
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # we've currently modified out ETH file on disk to need no mods
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        # preprocessing steps rename datetime column
        df_raw.rename(columns={'datetime':'date'}, inplace=True)
        df_raw.drop('utc_timestamp', axis=1, inplace=True)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols can be set to subset imported dataset right off the bat
        # cols will be input features ex. date and targets
        if self.cols:
            cols = self.cols.copy()
            cols = [x for x in cols if x not in self.targets]
            # cols.remove(self.target)
            print(cols)
            print('above is cols with target removed')
        else:
            cols = list(df_raw.columns);
            cols = [x for x in cols if x not in (self.targets+['date'])]
            print(cols)
            print('above is cols with target and date removed')
            # cols.remove(self.target);
            # cols.remove('date')
        # columns have be reordered to be date, features, target
        df_raw = df_raw[['date'] + cols + self.targets]
        print(df_raw)
        print('above is the first instance of df_raw, we are expecting a list of columns here')
        #num_train_bound = int(len(df_raw) * 0.7) - int(len(df_raw) * 0.7)%()
        num_train = int(len(df_raw) * 0.85)
        num_test = int(len(df_raw) * 0.075)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        print(border1s, border2s)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            if self.first_diff:
                print(df_data)
                df_data = df_data.diff().dropna()
            print(type(df_data.values[0][0]))
            print('df_data is df_raw[cols_data] to numpy feature vectors in pandas frame')
            print(df_data.values.mean())
        elif self.features == 'S':
            df_data = df_raw[self.targets]
            if self.first_diff:
                df_data = df_data.diff().dropna()

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            print(train_data.values)
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        print('calling time_features')
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        print('successfuling called time_features')

        # data_x here is truncating our at row borders separating train,val,test
        self.data_x = data[border1:border2]
        print('data_x is: ', self.data_x)

        xdims_dropped_for_pred = len(df_data.columns) - len(self.targets)
        # Whether to inverse output data, using this argument means inversing output data (defaults to False)
        # TODO implement inverse logic
        if self.inverse:
            # target columns are at end of df, here we drop everything else
            self.data_y = data[:,xdims_dropped_for_pred:].values[border1:border2]
        else:
            self.data_y = data[:,xdims_dropped_for_pred:][border1:border2]
        self.data_stamp = data_stamp
        print('verify dataset index set properly')
        print(len(self.data_x) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r denotes our y target values (commenting out label_len adjustment)
        r_begin = s_end + 1 # - self.label_len
        r_end = r_begin + self.pred_len # + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        # inverting the scaling of our inputs can be done with 'inverse_transform' method below
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        # commenting out cyclical calendar feature embedding generating vectors
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        #TODO
        # apply batch normalization here if we are
        # if normalize_batch:
        # check for scale flag to be set to false so we're not double normalizing
        return seq_x, seq_y, #seq_x_mark, seq_y_mark

    def __len__(self):
        # Each call requests a sample index for which the upperbound is specified in the __len__ method.
        return len(self.data_x) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
