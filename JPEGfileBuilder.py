import cv2
from tqdm import trange

def JPEGStorage():
    count = 0
    for iter in trange(2697):
        vidcap = cv2.VideoCapture('G:/open/train/TRAIN_' + str(iter).zfill(4) + '.mp4')
        success = True
        while success:
            success, image = vidcap.read()
            if success != True:
                break
            if count % 30 ==0:
                cv2.imwrite('G:/CompetitionCamera/data/'+str(int(count/30)) + '.jpg', image)
            count += 1

if __name__=="__main__":
    JPEGStorage()
