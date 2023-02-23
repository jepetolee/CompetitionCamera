import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import DCGN
import h5py
import warnings
from tqdm import trange
import random
import os
import cv2
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


def train_model(model, optimizer, train_loader, val_loader, scheduler, device):

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_val_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for videos,graph, labels in tqdm(iter(train_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output, aux_loss = model(videos.reshape(50,1024),graph[0], device)
            loss = criterion(output, labels)
            loss += aux_loss

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        _train_loss = np.mean(train_loss)
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 : [{_val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    preds, trues = [], []

    with torch.no_grad():
        for videos,graph, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)
            logit, aux_loss  = model(videos.reshape(50,1024),graph[0],device)

            loss = criterion(logit, labels) +aux_loss

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        _val_loss = np.mean(val_loss)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def dhash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, len(dhash_str), 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    return result


def hanming(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])

import torchvision.models as models
from KTS import kernel, cpd_auto2
class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list,saved=True,test=True):

        self.video_path_list = video_path_list
        self.label_list = label_list
        self.saved =saved

        if saved:
            if test:
                url = './my_data/test.h5'
            else:
                url = 'G:/CompetitionCamera/my_data/train.h5'
            self.f1 = h5py.File(url, 'r')

    def __getitem__(self, index):
        if self.saved:
            frames, graph = self.get_data(self.video_path_list[index])
        else:
            frames,graph = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames,graph, label
        else:
            return frames,graph

    def __len__(self):
        return len(self.video_path_list)

    def get_data(self,path):
        path = path[8:]
        return torch.from_numpy(self.f1[path]['features'][...].astype(np.float32)), self.f1[path]["change_points"][...].astype(np.int32)

    def get_video(self, path):

        googlenet = models.googlenet(pretrained=True)
        googlenet = torch.nn.Sequential(*list(googlenet.children())[:-2]).cuda()
        googlenet.eval()

        with torch.no_grad():
            frames = []
            cap = cv2.VideoCapture(path)
            hash1 = None
            base = None
            for i in range(CFG['VIDEO_LENGTH']):
                _, img = cap.read()
                hash2 = dhash(img)
                if hash1 is not None:
                    dist = hanming(hash1, hash2)
                if base is None or dist > 4:
                    base = img
                    hash1 = hash2
                    tensor = googlenet(torch.from_numpy(np.rollaxis(img, 2)).cuda().float().reshape(-1,3, 720, 1280))
                    frames.append(torch.squeeze(tensor))
                else:
                    tensor = googlenet(torch.from_numpy(np.rollaxis(base, 2)).cuda().float().reshape(-1,3, 720, 1280))
                    frames.append(torch.squeeze(tensor))
        tensor = torch.stack(frames,dim=0)
        K = tensor.cpu().detach().numpy()
        K1 = kernel(K, K.T, K.shape[0])
        m = round(50 / 106 * 2)
        cps1, scores1 = cpd_auto2(K1, m, 1.0, 1)
        cps1 *= 1
        cps1 = np.hstack((0, cps1, 50))
        begin_frames = cps1[:-1]
        end_frames = cps1[1:]
        cps1 = np.vstack((begin_frames, end_frames - 1)).T
        return tensor,cps1


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CFG = {
        'VIDEO_LENGTH': 50,
        'IMG_SIZE': 128,
        'EPOCHS': 100,
        'LEARNING_RATE': 0.005,
        'BATCH_SIZE': 1,
        'SEED': 41
    }
    seed_everything(CFG['SEED'])
    df = pd.read_csv('./train.csv')
    train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])

    train_dataset = CustomDataset(train['video_path'].values, train['label'].values,test=False)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val['video_path'].values, val['label'].values,test=False)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    model = DCGN.DCGN(1024, 13)
    check_point = torch.load('./saved.pt')
    model.load_state_dict(check_point)
    model.eval()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                           threshold_mode='abs', min_lr=1e-8, verbose=True)

    infer_model = train_model(model, optimizer, train_loader, val_loader, scheduler, device)

    test = pd.read_csv('./test.csv')
    test_dataset = CustomDataset(test['video_path'].values,None,test=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


    def inference(model, test_loader, device):
        model.to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for videos in tqdm(iter(test_loader)):
                videos = videos.to(device)

                logit = model(videos)

                preds += logit.argmax(1).detach().cpu().numpy().tolist()
        return preds


    preds = inference(infer_model, test_loader, device)

    submit = pd.read_csv('./sample_submission.csv')

    submit['label'] = preds

    submit.to_csv('./baseline_submit.csv', index=False)
