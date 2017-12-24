import torch
from imagedataset import ImageDataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from cnn import CNN
# from main import train_loader, batch_size
use_gpu = torch.cuda.is_available()
TRAIN_PATH = 'DL/data/processed/train.json'

train = pd.read_json(TRAIN_PATH)
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
train['band_1'] = train['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
train['band_2'] = train['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

batch_size = 10 #64
train_ds = ImageDataset(train[:10], include_target=True, u=0.5)
THREADS = 4
train_loader = DataLoader(train_ds, batch_size,
                               sampler=RandomSampler(train_ds),
                               num_workers=THREADS,
                               pin_memory=use_gpu)


img_size = (75,75)
img_ch = 2
kernel_size = 7
pool_size = 2
padding=2
n_out = 1
n_epoch = 35


if __name__ == '__main__':
    cnn = CNN(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out, padding=padding)
    cnn.fit(train_loader, n_epoch, batch_size, int(len(train_ds)))
