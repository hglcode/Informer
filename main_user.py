#!/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import torch
import numpy as np
import statistics as stat

from typing import Callable
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from loguru import logger

from models.model import Informer
from data_loader import STDataSet
from data_loader import FTDataSet
from early_stopper import EarlyStopper
from l_utils import LRScheduler

import torch._dynamo

torch._dynamo.config.suppress_errors = True

TORCH_COMPILE_DISABLED = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
EPOCHS = int(1e9)
LEARN_RATE = 1e-2

seq_len = 8
label_len = seq_len >> 1
pred_len = seq_len >> 1

dataset = FTDataSet((seq_len, label_len, pred_len), d_path = os.path.join(os.path.dirname(__file__), '.exchange/csv/utf8/c_2006_ft.csv'))
model = Informer(
    enc_in=dataset._data.shape[-1],
    dec_in=dataset._data.shape[-1],
    c_out=len(dataset.predict_indexs),
    seq_len=seq_len,
    label_len=label_len,
    out_len=pred_len,  # out_len,
    factor=5,
    d_model=512,
    n_heads=8,
    # e_layers must be equal to d_layers
    e_layers=64,
    d_layers=64,
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

try:
    pretrained = torch.load(os.path.join(os.path.dirname(__file__), '.out', 'informer.pth'))
    model.load_state_dict(pretrained['model'])
except:
    logger.warning('Failed load pretrained model.')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE, weight_decay=0.0)
lr_scheduler = LRScheduler(optimizer)


@torch.compile(disable=TORCH_COMPILE_DISABLED)
def train_one_batch(
    batch_x: torch.Tensor, batch_y: torch.Tensor, batch_x_mark: torch.Tensor, batch_y_mark: torch.Tensor, loss_fn: Callable
) -> torch.Tensor:
    batch_x = batch_x.float().to(DEVICE)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(DEVICE)
    batch_y_mark = batch_y_mark.float().to(DEVICE)

    # decoder input
    dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(DEVICE)
    # encoder - decoder
    with torch.autocast('cuda', torch.bfloat16):
        out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        lss = loss_fn(out, batch_y[:, -pred_len:, dataset.predict_indexs].to(DEVICE))
    # field:
    # f_dim = -1  # predict multiple value
    # batch_y = batch_y[:, -pred_len:, f_dim:].to(DEVICE)
    # batch_y = batch_y[:, -pred_len:, :C_OUT].to(DEVICE)
    return lss


@torch.compile(disable=TORCH_COMPILE_DISABLED)
def train() -> None:
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=0)
    train_loss = [0.0] * len(train_loader)
    best_model_dict = None
    best_score = float('inf')
    early_stop = EarlyStopper(8)
    for epoch in range(EPOCHS):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = train_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
            train_loss[i] = loss.item()
            loss.backward()
            optimizer.step()

        loss_avg = stat.mean(train_loss)
        lr_scheduler.step(loss_avg)
        if loss_avg < best_score:
            best_model_dict = copy.deepcopy(model.state_dict())
            best_score = loss_avg
        logger.info(f'{epoch:>2d}: loss: {loss_avg}, best: {best_score}, stop: {early_stop._counter}')
        if early_stop(loss_avg):
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
    logger.info(f'score: {best_score}, Save model to { model_path}!')


def valid_one_batch(
    batch_x: torch.Tensor, batch_y: torch.Tensor, batch_x_mark: torch.Tensor, batch_y_mark: torch.Tensor
) -> torch.Tensor:
    batch_x = batch_x.float().to(DEVICE)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(DEVICE)
    batch_y_mark = batch_y_mark.float().to(DEVICE)

    # decoder input
    dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(DEVICE)
    # encoder - decoder
    #out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #lss = criterion(out, batch_y[:, -pred_len:, :C_OUT].to(DEVICE))
    # field:
    # f_dim = -1  # predict multiple value
    # batch_y = batch_y[:, -pred_len:, f_dim:].to(DEVICE)
    # batch_y = batch_y[:, -pred_len:, :C_OUT].to(DEVICE)
    return model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

def valid() -> tuple[np.ndarray, np.ndarray]:
    model_path = './.out/informer.pth'
    pth = torch.load(model_path, weights_only=True)
    model.load_state_dict(pth['model'])
    model.eval()
    valid_loader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=False, num_workers=0)
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in valid_loader:
            pred = valid_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.cpu().numpy())
            trues.append(batch_y[:, -pred_len:, dataset.predict_indexs].cpu().numpy())

    #preds = np.concatenate(preds)
    #trues = np.concatenate(trues)
    #ps = dataset.scaler.inverse_transform(preds.reshape(-1, dataset._data.shape[-1]))
    #ts = dataset.scaler.inverse_transform(trues.reshape(-1, dataset._data.shape[-1]))
    #preds = ps.reshape(-1, preds.shape[-2], preds.shape[-1])
    #trues = ts.reshape(-1, trues.shape[-2], trues.shape[-1])
    return np.concatenate(preds), np.concatenate(trues)


if __name__ == '__main__':
    train()
    preds, trues = valid()
    logger.info(preds.shape)
    logger.info(trues.shape)
