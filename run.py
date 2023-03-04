import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import DCGN
import h5py
import warnings
import random
import os
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import albumentations as A
from skimage.util import random_noise


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)

            logit = model(videos, device)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds


class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list, test=True):
        self.test = test
        self.original = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']), A.Normalize()], p=1)
        self.flip = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
                               A.HorizontalFlip(p=1), A.Normalize()], p=1)
        self.bright = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
                                 A.RandomBrightness(limit=(-0.8, 0.8), p=1), A.Normalize()], p=1)
        self.blur = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
                               A.ElasticTransform(p=1), A.Normalize()], p=1)
        self.noise = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
                               A.GaussNoise(p=1), A.Normalize()], p=1)
        self.distort = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
                               A.GridDistortion(p=1), A.Normalize()], p=1)
        self.dropout = A.Compose([A.Resize(width=CFG['IMG_SIZE'], height=CFG['IMG_SIZE']),
                               A.CoarseDropout(p=1,max_holes=7,max_width=8,max_height=8,min_holes=3), A.Normalize()], p=1)
        self.video_path_list = video_path_list
        self.label_list = label_list

    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]

            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        with torch.no_grad():
            frames = []
            cap = cv2.VideoCapture(path)
            seed = random.randint(1, 100)
            if self.test is True:
                func = self.original
            elif 1 <= seed <= 30:
                func = self.original
            elif 31 <= seed <= 50:
                func = self.flip
            elif 51 <= seed <= 60:
                func = self.blur
            elif 61 <= seed <= 70:
                func = self.bright
            elif 71 <= seed <= 80:
                func = self.distort
            elif 81<= seed <= 90:
                func = self.dropout
            else:
                func = self.noise

            for i in range(CFG['VIDEO_LENGTH']):
                _, img = cap.read()
                img = func(image=np.asarray(img))['image']
                img = np.rollaxis(img, 2)
                frames.append(torch.from_numpy(img))
        return torch.stack(frames, dim=0).float()


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []
    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            logit = model(videos, device)
            labels = labels.to(device)
            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score


def train_model(model, optimizer, device, valid_mod=True):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_score = 0
    best_model = None
    preds, trues = [], []
    df = pd.read_csv('./train.csv')

    for epoch in range(1, CFG['EPOCHS'] + 1):
        train, val, _, _ = train_test_split(df, df['label'], test_size=0.2)

        train_dataset = CustomDataset(train['video_path'].values, train['label'].values, test=False)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

        val_dataset = CustomDataset(val['video_path'].values, val['label'].values, test=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
        model.train()

        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)

            optimizer.zero_grad()
            output = model(videos, device)
            labels = labels.to(device)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            pred = output.argmax(1)

            preds += pred.detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _train_loss = np.mean(train_loss)
        train_score = f1_score(trues, preds, average='macro')
        if valid_mod:
            _val_loss, _val_score = validation(model, criterion, val_loader, device)
            print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}], Train F1 : [{train_score:.5f}],Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')
            if best_score < _val_score:
                best_score = _val_score
                best_model = model
                torch.save(best_model.state_dict(), './saved.pt')
        else:

            print(
                f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Train F1 : [{train_score:.5f}]')
            if best_score < train_score:
                best_score = train_score
                best_model = model
                torch.save(best_model.state_dict(), './saved.pt')

    return best_model


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CFG = {
        'VIDEO_LENGTH': 50,
        'EPOCHS': 40,
        'IMG_SIZE': 256,
        'LEARNING_RATE': 0.0002,
        'BATCH_SIZE': 8
    }

    state_dict = torch.load('./saved.pt')
    model = DCGN.FinalModel().to(device)
    model.load_state_dict(state_dict)
    optimizer = torch.optim.RAdam(params=model.parameters(), lr=CFG["LEARNING_RATE"], weight_decay=1e-5)

    infer_model = train_model(model, optimizer, device)  # with valid

    test = pd.read_csv('./test.csv')
    test_dataset = CustomDataset(test['video_path'].values, None, test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    preds = inference(model, test_loader, device)

    submit = pd.read_csv('./sample_submission.csv')

    submit['label'] = preds

    submit.to_csv('./baseline_submit.csv', index=False)
