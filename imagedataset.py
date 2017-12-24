import numpy as np
import torch
import torch.utils.data as data
import cv2

class ImageDataset(data.Dataset):
    def __init__(self, X_data, include_target, u=0.5):
        """X_data = pandas df
        include_target - флаг. include_target = True если обучить и False если тестировать
        u - arg в X_transform функции
        """
        self.X_data = X_data
        self.include_target = include_target
        self.u = u

    def __getitem__(self, index):
        np.random.seed()
        # 2 канала нашего изобрадения
        img1 = self.X_data.iloc[index]['band_1']
        img2 = self.X_data.iloc[index]['band_2']

        # форма изображения (75,75,2)
        img = np.stack([img1, img2], axis=2)

        # получение углов и название изображения
        angle = self.X_data.iloc[index]['inc_angle']
        img_id = self.X_data.iloc[index]['id']

        # аргументация функции X_transform
        # if self.X_transform:
        img = self.random_vertical_flip(img, self.u)

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

    def random_vertical_flip(self, img, u=0.5):
        if np.random.random() < u:
            img2 = cv2.flip(img, 0)
            return img2
        return img