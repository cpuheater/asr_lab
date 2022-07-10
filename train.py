import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import math
import torch
import torch.nn as nn
from src.model import ResNet, BasicBlock
from src.dataset import SCDataset
import os
from src.utils import save_model, time_since, calc_f1, accuracy
from torchmetrics import F1Score
import torchaudio
import numpy as np

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=12, help='number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--grad-clip', default=5, help='gradient clipping value')
parser.add_argument('--model-dir', default='trained_models', help='dir saving model')
parser.add_argument('--hidden-size', default=64)
parser.add_argument('--data-dir', default="data", help='')

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = f'{os.path.basename(__file__).rstrip(".py")}_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}'
writer = SummaryWriter(f"runs/{exp_name}")


def train(epoch, train_loader, valid_loader, criterion, start, mel_spectogram):
    train_loss = 0
    train_accuracy = 0
    model.train()
    for i, (audio, label) in enumerate(train_loader, 1):
        audio, label = audio.to(device), label.to(device)
        spectogram = mel_spectogram(audio)
        pred = model(spectogram)
        loss = criterion(pred, label)
        train_loss += loss.data.item()
        model.zero_grad()
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        train_accuracy += accuracy(pred, label)

    print('[{}] Train Epoch: {} Loss: {:.2f} Accuracy: {:.6f}'.format(time_since(start),
                                                                      epoch, train_loss, 100 * train_accuracy/len(train_loader.dataset)))
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/accuracy", 100 * train_accuracy/len(train_loader.dataset), epoch)
    valid_loss = 0
    valid_accuracy = 0
    valid_epoch_pred = []
    valid_epoch_target = []
    model.eval()
    with torch.no_grad():
        for i, (audio, label) in enumerate(valid_loader, 1):
            audio, label = audio.to(device), label.to(device)
            spectogram = mel_spectogram(audio)
            pred = model(spectogram)
            loss = criterion(pred, label)
            valid_loss += loss.data.item()
            valid_accuracy += accuracy(pred, label)
            valid_epoch_pred.append(pred)
            #valid_epoch_target.append(label)

    #valid_f1 = calc_f1(torch.cat(valid_epoch_pred), torch.cat(valid_epoch_target))
    writer.add_scalar("valid/loss", valid_loss, epoch)
    writer.add_scalar("valid/accuracy", 100 * valid_accuracy/len(valid_loader.dataset), epoch)
    #writer.add_scalar("valid/F1", valid_f1, epoch)
    print('[{}] Valid Epoch: {} Loss: {:.6f} Accuracy: {:.2f} F1: :.6f'.format(time_since(start), epoch, valid_loss,
                                                                                 100 * valid_accuracy/len(valid_loader.dataset), 0))
    return

def collate_fn(batch, labels_list):
    audios, labels = [], []
    for audio, _, label, *_ in batch:
        audios.append(audio.flatten())
        labels.append(torch.tensor(labels_list.index(label)))

    audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.)
    labels = torch.stack(labels)
    audios_padded = audios_padded.unsqueeze(1)
    return audios_padded, labels



if __name__ == '__main__':
    train_dataset = SCDataset("training")
    valid_dataset = SCDataset("testing")
    labels = sorted(list(set(datapoint[2] for datapoint in train_dataset)))

    sampling_rate = next(iter(train_dataset))[1]
    n_fft = int(np.floor(sampling_rate/10))
    hop_length = int(np.floor(sampling_rate/40))
    n_mels = 128
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate,
                                                                         n_fft=n_fft,
                                                                         hop_length=hop_length,
                                                                         n_mels=n_mels).to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True,  collate_fn=lambda batch: collate_fn(batch, labels))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                               shuffle=True, collate_fn=lambda batch: collate_fn(batch, labels))

    model = ResNet(BasicBlock, layers=[3, 4, 6, 3], num_classes=len(labels)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    start = time.time()
    print("Training for %d epochs..." % args.epochs)
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader, valid_loader, criterion, start, mel_spectrogram)

    save_model(model)
