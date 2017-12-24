import torch
from cnn import CNN
import pandas as pd
from imagedataset import ImageDataset
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()
TEST_PATH = 'DL/data/processed/test.json'

img_size = (75,75)
img_ch = 2
kernel_size = 7
pool_size = 2
padding=2
n_out = 1
n_epoch = 35
batch_size = 10
THREADS = 4

model = CNN(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size, pool_size=pool_size, n_out=n_out, padding=padding)
model.load_state_dict(torch.load('cnn.pth'))
print(model)

test = pd.read_json(TEST_PATH)
test['inc_angle'] = pd.to_numeric(test['inc_angle'], errors='coerce')
test['band_1'] = test['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
test['band_2'] = test['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
test_ds = ImageDataset(test, include_target=False, u=0.5)
test_loader = DataLoader(test_ds, batch_size,
                               sampler=RandomSampler(test_ds),
                               num_workers=THREADS,
                               pin_memory=use_gpu)
test.head(3)

print(test.shape)
columns = ['id', 'is_iceberg']
df_pred = pd.DataFrame(data=np.zeros((0, len(columns))), columns=columns)
# df_pred.id.astype(int)

for index, row in test.iterrows():
    rwo_no_id = row.drop('id')
    band_1_test = (rwo_no_id['band_1']).reshape(-1, 75, 75)
    band_2_test = (rwo_no_id['band_2']).reshape(-1, 75, 75)
    full_img_test = np.stack([band_1_test, band_2_test], axis=1)

    x_data_np = np.array(full_img_test, dtype=np.float32)
    if use_gpu:
        X_tensor_test = Variable(torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
    else:
        X_tensor_test = Variable(torch.from_numpy(x_data_np))  # Note the conversion for pytorch

    predicted_val = (model(X_tensor_test).data).float()  # probabilities
    p_test = predicted_val.cpu().numpy().item()  # otherwise we get an array, we need a single float

    df_pred = df_pred.append({'id': row['id'], 'is_iceberg': p_test}, ignore_index=True)

df_pred.head(5)

def savePred(df_pred):
    csv_path = 'sample_submission.csv'
    df_pred.to_csv(csv_path, columns=('id', 'is_iceberg'), index=None)
    print(csv_path)

savePred(df_pred)