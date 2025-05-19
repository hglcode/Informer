#!/bin/env python3
# -*- coding: utf-8 -*-
import copy
import torch
import numpy as np
import statistics as stat
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from models.model import Informer
from data_loader import STDataSet
from data_loader import FTDataSet

TORCH_COMPILE_DISABLED = True

class MyLR:
    def __init__(self, optimizer: optim.Adam, step_rate: float = 0.9, threshold: int = 1) -> None:
        self._optimizer = optimizer
        self._last_lr = [group['lr'] for group in self._optimizer.param_groups]
        self._best_score = float('inf')
        self._last_score = float('inf')
        self._best_steps = 0
        self._step_rate = step_rate
        self._threshold = threshold

    def step(self, score: float) -> None:
        '''
        if score < self._best_score:
            self._best_score = score
            self._best_steps = 0
            return
        self._best_steps += 1
        if self._best_steps < self._threshold:
            return
        self._best_steps = 0
        '''
        last_score = self._last_score
        self._last_score = score
        if score <= last_score:
            return
        self._last_lr = [lr * self._step_rate for lr in self._last_lr]
        for i, group in enumerate(self._optimizer.param_groups):
            group['lr'] = self._last_lr[i]
        print(f'Update lr to: {self._last_lr}')


class EarlyStop(object):
    def __init__(self, patience: int = 18) -> None:
        self._patience: int = patience
        self._counter: int = 0
        self._best_sore: float = float('inf')
        self._train_loss: float = float('inf')
        self._valid_loss: float = float('inf')

    def stop(self, valid_loss: float) -> bool:
        if valid_loss < self._best_sore:
            self._best_sore = valid_loss
            self._counter = 0
            return False

        self._counter += 1
        return self._counter > self._patience


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = int(1e9)

seq_len = 20
label_len = 20
pred_len = 5

dataset = FTDataSet((seq_len, label_len, pred_len))

enc_in = dataset[0][0].shape[-1]
dec_in = enc_in
c_out = 2  # enc_in  # 4

# e_layers > 5 and d_layers > 4 will encountered error
e_layers = 8
d_layers = 8

model = Informer(
    enc_in,
    dec_in,
    c_out,
    seq_len,
    label_len,
    pred_len,  # out_len,
    factor=5,
    d_model=512,
    n_heads=8,
    e_layers=e_layers,
    d_layers=d_layers,
    d_ff=512,
    dropout=0.0,
    attn='full',  # 'prob',
    embed='timeF',  # 'fixed',
    freq='b',
    activation='gelu',
    output_attention=False,
    distil=True,
    mix=True,
    device=DEVICE,
).to(DEVICE)

model_lr = 1e-1
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=model_lr, weight_decay=0.0)
lr_scheduler = MyLR(optimizer)


@torch.compile(disable=TORCH_COMPILE_DISABLED)
def process_one_batch(
    batch_x: torch.Tensor, batch_y: torch.Tensor, batch_x_mark: torch.Tensor, batch_y_mark: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_x = batch_x.float().to(DEVICE)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(DEVICE)
    batch_y_mark = batch_y_mark.float().to(DEVICE)

    # decoder input
    dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(DEVICE)
    # encoder - decoder
    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    # field:
    # f_dim = -1  # predict multiple value
    # batch_y = batch_y[:, -pred_len:, f_dim:].to(DEVICE)
    batch_y = batch_y[:, -pred_len:, :c_out].to(DEVICE)

    return outputs, batch_y


@torch.compile(disable=TORCH_COMPILE_DISABLED)
def train() -> None:
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=0)
    train_loss = [0.0] * len(train_loader)
    best_model_dict = None
    best_score = float('inf')
    early_stop = EarlyStop(8)
    for epoch in range(EPOCHS):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            pred, true = process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss: torch.Tensor = criterion(pred, true)
            train_loss[i] = loss.item()
            loss.backward()
            optimizer.step()

        loss_avg = stat.mean(train_loss)
        lr_scheduler.step(loss_avg)
        if loss_avg < best_score:
            best_model_dict = copy.deepcopy(model.state_dict())
            best_score = loss_avg
        print(f'{epoch:>2d}: loss: {loss_avg}, best: {best_score}, stop: {early_stop._counter}')
        if early_stop.stop(loss_avg):
            break

    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
    else:
        raise RuntimeError("No best model found to load.")

    save_data = {
        'model': best_model_dict,
        'optimizer': optimizer.state_dict(),
    }
    model_path = './.out/informer.pth'
    torch.save(save_data, model_path)
    print(f'score: {best_score}, Save model to { model_path}!')


def valid() -> tuple[np.ndarray, np.ndarray]:
    model_path = './.out/informer.pth'
    pth = torch.load(model_path, weights_only=True)
    model.load_state_dict(pth['model'])
    model.eval()
    valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=0)
    pred_list = []
    true_list = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in valid_loader:
            pred, true = process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            pred_list.append(pred.cpu().numpy())
            true_list.append(true.cpu().numpy())

    preds = np.concatenate(pred_list)
    trues = np.concatenate(true_list)
    ps = dataset.scaler.inverse_transform(preds.reshape(-1, dataset._data.shape[-1]))
    ts = dataset.scaler.inverse_transform(trues.reshape(-1, dataset._data.shape[-1]))
    preds = ps.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = ts.reshape(-1, trues.shape[-2], trues.shape[-1])
    return preds, trues


if __name__ == '__main__':
    #train()
    preds, trues = valid()
    print(preds.shape)
    print(trues.shape)
