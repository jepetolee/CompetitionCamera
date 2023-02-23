import torch
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import cv2
import os
import h5py
from time import time

EXTRACT_FOLDER = './test'

EXTRACT_FREQUENCY = 1
BATCH_SIZE = 16

googlenet = models.googlenet(pretrained=True)
googlenet = torch.nn.Sequential(*list(googlenet.children())[:-2])
googlenet.eval()
googlenet = googlenet.cuda()


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
    # print(result)
    return result


def hanming(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


def one_video_hash(path, threshold):
    cap = cv2.VideoCapture(path)
    frames = []
    video_features = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    skip_count = 0
    arr = []

    with torch.no_grad():
        base = None
        hash1 = None
        while cap.isOpened():
            # Capture frame-by-frame
            ret, fr = cap.read()
            if ret is False:
                break
            count += 1

            if count % EXTRACT_FREQUENCY == 0:
                hash2 = dhash(fr)
                if hash1 is not None:
                    dist = hanming(hash1, hash2)
                if base is None or dist > threshold:
                    base = fr
                    hash1 = hash2
                    frames.append(np.rollaxis(fr, 2))
                    arr.append(skip_count)
                    skip_count = 0
                else:
                    skip_count += 1
                    frames.append(np.rollaxis(base, 2))
                if (len(frames) == BATCH_SIZE) or \
                        (count == frame_count // EXTRACT_FREQUENCY * EXTRACT_FREQUENCY and len(frames) > 0):
                    batch = np.array(frames)

                    variable = Variable(torch.from_numpy(batch).float()).cuda()
                    feature = googlenet(variable).cpu().detach().numpy()
                    video_features.extend(feature)
                    frames.clear()

    cap.release()
    video_features = np.squeeze(np.array(video_features))
    return video_features, frame_count, fps, arr


def one_video(path):
    cap = cv2.VideoCapture(path)
    count = 0
    frames = []
    video_features = []
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    with torch.no_grad():
        while cap.isOpened():
            # Capture frame-by-frame
            ret, fr = cap.read()
            if ret is False:
                break
            count += 1
            if count % EXTRACT_FREQUENCY == 0:
                frames.append(np.rollaxis(fr, 2))
            if (len(frames) == BATCH_SIZE) or \
                    (count == frame_count // EXTRACT_FREQUENCY * EXTRACT_FREQUENCY and len(frames) > 0):
                batch = np.array(frames)

                variable = Variable(torch.from_numpy(batch).float()).cuda()
                feature = googlenet(variable).cpu().detach().numpy()
                video_features.extend(feature)
                frames.clear()
    cap.release()

    video_features = np.squeeze(np.array(video_features))

    return video_features, frame_count, fps


h5_file = 'my_data/my_data'
f = h5py.File(h5_file, 'w')
files = os.listdir(EXTRACT_FOLDER)
cnt = 0

for file in files:
    cnt += 1
    path = EXTRACT_FOLDER + "/" + file
    st = time()
    video_features, fcnt, fps, skp_arr = one_video_hash(path, 4.0)

    duration = fcnt / fps
    ed = time()

    print(cnt, file, ed - st, fcnt, video_features.shape)

    f.create_dataset(file + '/n_frames', data=int(fcnt))
    f.create_dataset(file + '/features', data=video_features)
    picks = np.arange(0, video_features.shape[0]) * EXTRACT_FREQUENCY
    f.create_dataset(file + '/picks', data=picks)
    f.create_dataset(file + '/time1', data=ed - st)
    f.create_dataset(file + '/duration', data=duration)

    f.create_dataset(file + '/skip_arr', data=skp_arr)

f.close()
