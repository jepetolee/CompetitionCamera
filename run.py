import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import DCGN
import h5py
import warnings
from tqdm import trange


def train(model, optimizer, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    best_loss = 99999
    best_model = None
    file_name = './my_data/train.h5'
    df = pd.read_csv('./train.csv')
    label = df['label']

    f1 = h5py.File(file_name, 'r')

    for epoch in range(CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []

        for i in trange(2698):
            link = 'TRAIN_' + str(i).zfill(4) + '.mp4'
            graph = f1[link]['change_points'][...].astype(np.int32)
            videos = torch.from_numpy(f1[link]['features'][...].astype(np.float32)).to(device)

            labels = torch.zeros([1, 13]).to(device)
            labels[0][label[i]] += 1

            optimizer.zero_grad()
            output = model(videos, graph, device)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}]')

        if scheduler is not None:
            scheduler.step(_train_loss)

        if best_loss > _train_loss:
            best_loss = _train_loss
            best_model = model
            torch.save(best_model.state_dict(), 'saved.pt')

    return best_model


def prediction(model, device):
    model.eval()
    predictions = []

    file_name = './my_data/test.h5'
    f1 = h5py.File(file_name, 'r')

    with torch.no_grad():
        for i in trange(1800):
            link = 'TEST_' + str(i).zfill(4) + '.mp4'
            graph = f1[link]["change_points"][...].astype(np.int32)
            videos = torch.from_numpy(f1[link]['features'][...].astype(np.float32)).to(device)
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos, graph, device)
            predictions += logit.argmax(1).detach().cpu().numpy().tolist()
    submit = pd.read_csv('./sample_submission.csv')
    submit['label'] = predictions
    submit.to_csv('./baseline_submit.csv', index=False)
    return


warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'VIDEO_LENGTH': 50,
    'IMG_SIZE': 256,
    'EPOCHS': 30,
    'LEARNING_RATE': 0.01,
    'BATCH_SIZE': 1,
    'SEED': 41
}

model = DCGN.DCGN(1024, 13)
# check_point = torch.load('./saved.pt')
# model.load_state_dict(check_point)
model.eval()
optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                       threshold_mode='abs', min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, scheduler, device)
prediction(infer_model, device)
