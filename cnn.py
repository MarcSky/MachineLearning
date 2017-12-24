import torch
import torch.nn as nn
use_gpu = torch.cuda.is_available()

def cnnBlock(in_planes, out_planes, kernel_size=7, padding=2, pool_size=2):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding)
    bn = torch.nn.BatchNorm2d(out_planes)
    relu = torch.nn.LeakyReLU()
    pl = torch.nn.MaxPool2d(pool_size, pool_size)
    av = torch.nn.AvgPool2d(pool_size, pool_size)
    # dr   = torch.nn.Dropout(d_rate)
    return nn.Sequential(conv, bn, relu, pl, av)


class CNN(nn.Module):
    def __init__(self, img_size, img_ch, kernel_size, pool_size, n_out, padding):
        super(CNN, self).__init__()
        self.img_size = img_size
        self.img_ch = img_ch
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding

        self.n_out = n_out
        self.sig = torch.nn.Sigmoid()
        self.all_losses = []
        self.val_losses = []
        self.cnn_features = []
        self.layers = []
        self.build_model()

    def build_model(self):
        self.conv1 = cnnBlock(self.img_ch, 16, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = cnnBlock(16, 32, kernel_size=5, padding=self.padding)
        self.conv3 = cnnBlock(32, 64, kernel_size=3, padding=self.padding)

        self.cnn_features = [self.conv1, self.conv2, self.conv3]
        self.fc = nn.Sequential(nn.Linear(64, self.n_out))
        self.criterion = torch.nn.BCELoss()
        LR = 0.0005
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LR, weight_decay=5e-5)  # L2 regularization

    def forward(self, x):
        for c in self.cnn_features:
            x = (c(x))
        x = self.shrink(x)
        x = self.fc(x)
        return self.sig(x)

    def shrink(self, X):
        return X.view(X.size(), -1)

    def fit(self, loader, num_epochs, batch_size, train_ds_size = 0):
        self.train()
        print('START TRAINING')
        for epoch in range(num_epochs):
            self.train()
            print('Epoch {}'.format(epoch + 1))
            print('*' * 5 + ':')
            # running_loss = 0.0

            for i, dict_ in enumerate(loader):
                images = dict_['img']
                target = dict_['target']#.type(torch.FloatTensor)
                if use_gpu:
                    images = images.cuda()
                    target = target.cuda()

                inputs = torch.autograd.Variable(images)
                labels = torch.autograd.Variable(target)

                preds = self.forward(inputs)  # cnn выход
                loss = self.criterion(preds, labels)  # функция потерь кроссэнтропия
                # running_loss += loss.data[0] * labels.size(0)
                self.optimizer.zero_grad()  # чистим градиент на каждом шаге
                loss.backward()  # обратное распространие ошибки, градиентный спуск
                self.optimizer.step()  # вставляем градиенты
                preds = torch.max(preds, 1)[1].data.numpy().squeeze()
                acc = (preds == target.numpy()).mean()
                if (i + 1) % 10 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                          % (epoch + 1, num_epochs, i + 1, int(train_ds_size / batch_size), loss.data[0], acc))

        torch.save(self.state_dict(), 'cnn.pth')