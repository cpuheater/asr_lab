
import time
import math
import torch
import os
from torchmetrics import F1Score

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def accuracy(pred, labels):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == labels.squeeze()).item()

def save_model(model, exp_name, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(),
               os.path.join(dir, f'{exp_name}.pt'))

def calc_f1(pred, target, device):
    f1 = F1Score(1).to(device)
    pred = torch.round(pred.squeeze())
    return f1(pred, target)
