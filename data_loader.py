import os
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler  # type: ignore
from torch.utils.data import Dataset
from utils.timefeatures import time_features

'''
    columns:
    re_cols = [
        'low',
        'high',
        'open',
        'close',
        'volume',
        'rise',
        'amplitude',
        'turnover',
        'hld_rate',
    ]
    data = data.reindex(columns=re_cols)
'''

COLUMNS = {
    '时间': 'time',
    '开盘': 'open',
    '最高': 'high',
    '最低': 'low',
    '收盘': 'close',
    '涨幅': 'rise',
    '振幅': 'amplitude',
    '总手': 'volume',
    '金额': 'turnover',
    '换手': 'hld_rate',
}

DTYPES_ST = {
    '时间': str,
    '开盘': float,
    '最高': float,
    '最低': float,
    '收盘': float,
    '涨幅': float,
    '振幅': float,
    '总手': int,
    '金额': float,
    '换手': float,
}
DTYPES_FT = {
    'code': str,
    #'date': str,
    'open': float,
    'high': float,
    'low': float,
    'close': float,
    'balance': float,
    'up_dn_1': float,
    'up_dn_2': float,
    'volume': int,
    'turnover': float,
    'position': int,
    'year': int,
    'month': int,
    'day': int,
    'week': int,
    'wday': int,
    'yday': int,
    'mdays': int,
    'weekend': int,
    'month_b': int,
    'month_e': int,
    'quarter_b': int,
    'quarter_e': int,
    'year_b': int,
    'year_e': int,
    'leap_year': int,
    'prev': int,
    'next': int,
}


class STDataSet(Dataset):
    def __init__(
        self,
        sizes: tuple[int, int, int],  # (seq_len, label_len, pred_len)
        data_path: str = '~/.data/stock/debug/002594_day.csv',
    ) -> None:
        super().__init__()
        self.scaler: StandardScaler = StandardScaler()
        self._data_path: str = os.path.expanduser(data_path)
        self._seq_len, self._label_len, self._pred_len = sizes
        self._read_data()

    def _read_data(self) -> None:
        df = pd.read_csv(
            self._data_path,
            parse_dates=['date'],
            date_format='%Y/%m/%d',
            dtype=DTYPES_ST,
        )
        # 去掉第一条数据，原因是：第一条数据涨幅和振幅为空
        df = df[1:]
        df.rename(columns=COLUMNS, inplace=True)
        df = df[df['turnover'] != '--']
        df['turnover'] = df['turnover'].astype(np.int64)
        # old code:
        # df['rise_amp'] = df['rise_amp'].apply(lambda x: float(x[: -1 if x.endswith('%') else len(x)]))
        # df['amplitude'] = df['amplitude'].apply(lambda x: float(x[2:-1]))
        # df['hld_rate'] = df['hld_rate'].apply(lambda x: float(x[:-1]))
        # stamp = pd.DataFrame({'date': df.time})
        # data = df.drop(columns=['time'])
        stamp = df[['date']]
        data = df.drop(columns=['date'])
        '''
        rcols = [
            'low',
            'high',
            'open',
            'close',
            'volume',
            'rise',
            'amplitude',
            'turnover',
            'hld_rate',
        ]
        data = data.reindex(columns=rcols)
        '''
        self.columns = list(data.columns)
        self.scaler.fit(data.values)
        stamp_features = time_features(stamp, 1, 'b')
        if stamp_features is None:
            raise ValueError("time_features returned None, cannot assign to self._stamp")
        self._stamp = stamp_features
        self._data = self.scaler.transform(data.values)
        self._data_frame = df

    def __getitem__(self, idx: int) -> tuple:
        xa = idx
        xb = xa + self._seq_len
        ya = xb - self._label_len
        yb = ya + self._label_len + self._pred_len

        x = self._data[xa:xb]
        y = self._data[ya:yb]
        x_mark = self._stamp[xa:xb]
        y_mark = self._stamp[ya:yb]

        return x, y, x_mark, y_mark

    def __len__(self) -> int:
        return len(self._data) - (self._seq_len + self._pred_len)


class FTDataSet(Dataset):

    def __init__(
        self,
        sizes: tuple[int, int, int],
        data_path: str = os.path.join(os.path.dirname(__file__), '.exchange/csv/utf8/c_2006_ft.csv'),
    ) -> None:
        super().__init__()
        self.scaler: StandardScaler = StandardScaler()
        self._data_path: str = os.path.expanduser(data_path)
        self._seq_len, self._label_len, self._pred_len = sizes
        self._read_data()

    def _read_data(self) -> None:
        df = pd.read_csv(self._data_path, parse_dates=['date'], date_format='%Y%m%d', dtype=DTYPES_FT)
        # 去掉第一条数据，原因是：第一条数据涨幅和振幅为空
        df = df[1:]
        df['code'] = df['code'].apply(lambda x: re.sub(r'^[a-z]+', '', x, flags=re.IGNORECASE))
        df['code1'] = df['code'].apply(lambda x: int(x[:2]))
        df['code2'] = df['code'].apply(lambda x: int(x[2:]))
        df.drop(columns=['code'], inplace=True)
        stamp = df[['date']]
        data = df.drop(columns=['date'])
        self.columns = list(data.columns)
        self.scaler.fit(data.values)
        stamp_features = time_features(stamp, 1, 'b')
        if stamp_features is None:
            raise ValueError("time_features returned None, cannot assign to self._stamp")
        self._stamp = stamp_features
        self._data = self.scaler.transform(data.values)
        self._data_frame = df

    def __getitem__(self, idx: int) -> tuple:
        xa = idx
        xb = xa + self._seq_len
        ya = xb - self._label_len
        yb = ya + self._label_len + self._pred_len

        x = self._data[xa:xb]
        y = self._data[ya:yb]
        x_mark = self._stamp[xa:xb]
        y_mark = self._stamp[ya:yb]

        return x, y, x_mark, y_mark

    def __len__(self) -> int:
        return len(self._data) - (self._seq_len + self._pred_len)
