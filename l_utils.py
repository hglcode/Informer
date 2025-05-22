import os
import math
import torch
import pandas as pd
import torch.distributed as dist
import torch.optim as optim

from loguru import logger
from types import TracebackType
from typing import Any, Optional, Type
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime, time

from f_utils.datetime import is_future_cn_trading_time  # type: ignore


def _get_invalid_index_list(df: pd.DataFrame, price_max: float) -> list:
    idx_lst = []
    for i, r in df.iterrows():
        if r.close > 0 or not is_future_cn_trading_time(r.timestamp) or r.ask_p1 > price_max or r.bid_p1 > price_max:
            idx_lst.append(i)
    return idx_lst


def remove_dirty_data(df: pd.DataFrame, price_max: float, inplace: bool = False) -> pd.DataFrame | None:
    return df.drop(labels=_get_invalid_index_list(df, price_max), axis=0, inplace=inplace)


def _group_apply(x: pd.DataFrame) -> pd.Series:
    d = x.iloc[0]
    l = len(x)

    d['open'] = x.iloc[0].last
    d['high'] = x['last'].max()
    d['low'] = x['last'].min()
    d['close'] = x.iloc[-1].last

    d.volume = (x.volume / l).sum()
    d.high = x.high.max()
    d.low = x.low.min()
    d.average = (x.average / l).sum()
    d.ask_p1 = (x.ask_p1 / l).sum()
    d.ask_v1 = (x.ask_v1 / l).sum()
    d.bid_p1 = (x.bid_p1 / l).sum()
    d.bid_v1 = (x.bid_v1 / l).sum()
    d.position = (x.position / l).sum()
    d.turnover = (x.turnover / l).sum()

    return d


def tick2second(df: pd.DataFrame) -> pd.DataFrame:
    def _apply(x: pd.Timestamp) -> datetime:
        t = x.time()
        return datetime.combine(x.date(), time(t.hour, t.minute, t.second))

    df['dt'] = df.timestamp.apply(_apply)
    return df.groupby(by='dt', as_index=False, sort=True, dropna=True)[df.columns].apply(_group_apply)


def tick2minute(df: pd.DataFrame) -> pd.DataFrame:
    df['dt'] = df.timestamp.apply(lambda x: datetime.combine(x.date(), time(x.time().hour, x.time().minute)))
    return df.groupby(by='dt', as_index=False, sort=True, dropna=True)[df.columns].apply(_group_apply)


def build_model_name(loss: float | None = None, score: float | None = None, prefix: str = 'model', desc: str = '') -> str:
    time = datetime.now().strftime('%Y%m%d%H%M%S')
    if loss is None or score is None:
        return f'{prefix}_{desc}_{time}.pth'
    best_loss, best_score = loss, math.fabs(score)
    return f'{prefix}_{desc}_{time}_{best_loss}_{best_score}.pth'.replace('__', '_')


def _get_tvt_len(dataset: Any, train: float = 0.9, test: bool = False) -> tuple[int, int] | tuple[int, int, int]:
    train_len = int(len(dataset) * train)
    if not test:
        valid_len = len(dataset) - train_len
        return max(train_len, valid_len), min(train_len, valid_len)
    valid_len = math.ceil((len(dataset) - train_len) / 2)
    test_len = len(dataset) - train_len - valid_len
    return train_len, valid_len, test_len


def build_data_loader(
    dataset: Any, rank: int, batch_size: int, n_gpu: int = torch.cuda.device_count(), shuffle: bool = True
) -> DataLoader:
    worker_num = (os.cpu_count() or 0) // n_gpu
    sampler = DistributedSampler(dataset, num_replicas=n_gpu, rank=rank, shuffle=shuffle, drop_last=False) if n_gpu > 1 else None
    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=worker_num,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )


def random_split_dataset(
    dataset: Any,
    rank: int,
    batch_size: int,
    n_gpu: int = torch.cuda.device_count(),
    train_ratio: float = 0.98,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_set, valid_set = random_split(dataset, _get_tvt_len(dataset, train_ratio, False))
    train_loader = build_data_loader(train_set, rank, batch_size, n_gpu, shuffle)
    valid_loader = build_data_loader(valid_set, rank, batch_size, n_gpu, shuffle)
    return train_loader, valid_loader


class DDPContext:
    '''
    # 主节点的 IP 地址和端口号
    master_addr = '192.168.1.100'
    master_port = '12345'
    init_method = f'tcp://{master_addr}:{master_port}'
    '''

    def __init__(self, rank: int, n_gpu: int, backend: str = "nccl", init_method: str = 'tcp://127.0.0.1:65535') -> None:
        self._rank = rank
        self._world_size = n_gpu
        self._backend = backend
        self._init_method = init_method

    def __enter__(self) -> Any:
        dist.init_process_group(self._backend, rank=self._rank, world_size=self._world_size, init_method=self._init_method)
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        dist.destroy_process_group()


class LRScheduler:
    def __init__(self, optimizer: optim.Optimizer, step_rate: float = 0.8, threshold: int = 3) -> None:
        self._optimizer = optimizer
        self._last_lr = [group['lr'] for group in self._optimizer.param_groups]
        self._best_score = float('inf')
        self._best_steps = 0
        self._step_rate = step_rate
        self._threshold = threshold

    def step(self, score: float) -> None:
        if score < self._best_score:
            self._best_score = score
            self._best_steps = 0
            return
        self._best_steps += 1
        if self._best_steps < self._threshold:
            return

        self._best_score = float('inf')
        self._best_steps = 0
        self._last_lr = [lr * self._step_rate for lr in self._last_lr]
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = self._last_lr[i]
        logger.info(f'Update lr to: {self._last_lr}')

    def update(self, score: float) -> None:
        self.step(score)

    @property
    def learn_rate(self) -> list[float]:
        return self._last_lr
