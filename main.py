import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
# from torch.autograd import Variable
from torch.utils.data.sampler import RandomSampler
import cv2

TRAIN_PATH = 'DL/data/processed/train.json'
TEST_PATH = 'DL/data/processed/test.json'

class ImageDataset(data.Dataset):
    def __init__(self, X_data, include_target, u=0.5, X_transform=None):
        """X_data = pandas df
        include_target - флаг. include_target = True если обучить и False если тестировать
        u - arg в X_transform функции
        X_transform - аргумент функции
        """
        self.X_data = X_data
        self.include_target = include_target
        self.X_transform = X_transform
        self.u = u

    def __getitem__(self, index):
        np.random.seed()
        # 2 канала нашего изобрадения
        img1 = self.X_data.iloc[index]['band_1']
        img2 = self.X_data.iloc[index]['band_2']

        # форма изображения (75,75,2)
        img = np.stack([img1, img2], axis=2)

        # получение углов и название изображения
        if self.include_target == True:
            angle = self.X_data.iloc[index]['inc_angle']
        else:
            angle = None
        img_id = self.X_data.iloc[index]['id']

        # аргументация функции X_transform
        if self.X_transform:
            img = self.X_transform(img, **{'u': self.u})

        # меняем форму для pytorch
        img = img.transpose((2, 0, 1))
        img_numpy = img.astype(np.float32)
        # преобразование numpy в tensor
        img_torch = torch.from_numpy(img_numpy)

        dict_ = {'img': img_torch,
                 'id': img_id,
                 'angle': angle,
                 'img_np': img_numpy}

        if self.include_target:
            target = self.X_data.iloc[index]['is_iceberg']
            dict_['target'] = target
        return dict_

    def __len__(self):
        return len(self.X_data)


#поворот изображения вокруг оси X
def random_vertical_flip(img, u=0.5):
    if np.random.random() < u:
        img = cv2.flip(img, 0)
    return img

train = pd.read_json(TRAIN_PATH)
train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors='coerce')
train['band_1'] = train['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
train['band_2'] = train['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

batch_size = 10 #64
train_ds = ImageDataset(train, include_target=True, u=0.5, X_transform=random_vertical_flip)
USE_CUDA = False
THREADS = 4
train_loader = data.DataLoader(train_ds, batch_size,
                               sampler=RandomSampler(train_ds),
                               num_workers=THREADS,
                               pin_memory=USE_CUDA)


test = pd.read_json(TEST_PATH)
test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
test_ds = ImageDataset(train, include_target=False, u=0.5, X_transform=random_vertical_flip)
test_loader = data.DataLoader(test_ds, batch_size,
                               sampler=RandomSampler(test_ds),
                               num_workers=THREADS,
                               pin_memory=USE_CUDA)


# for i, dict_ in enumerate(train_loader):
#     images = dict_['img'] #изображения
#     target = dict_['target'].type(torch.FloatTensor) #это корабль или айсберг
#
#     if USE_CUDA:
#         images = images.cuda()
#         target = target.cuda()
#
#     images = Variable(images)
#     target = Variable(target)
#
#     # tобучение
#     # prediction = Net().forward(images)
#     # ....
#
#     # для ядра
#     print(target)
#     if i == 0:
#         break