import torch
import torch.nn as nn
from main import train_ds

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
        return X.view(X.size(0), -1)

    def fit(self, loader, num_epochs, batch_size):
        self.train()
        for epoch in range(num_epochs):
            self.train()
            print('Epoch {}'.format(epoch + 1))
            print('*' * 5 + ':')
            running_loss = 0.0
            # running_acc = 0.0

            for i, dict_ in enumerate(loader):
                images = dict_['img']
                target = dict_['target']
                inputs = torch.autograd.Variable(images)
                labels = torch.autograd.Variable(target)

                preds = self.forward(inputs)  # cnn выход
                loss = self.criterion(preds, labels)  # функция потерь кроссэнтропия
                running_loss += loss.data[0] * labels.size(0)
                self.optimizer.zero_grad()  # чистим градиент на каждом шаге
                loss.backward()  # обратное распространие ошибки, градиентный спуск
                self.optimizer.step()  # вставляем градиенты
                preds = torch.max(preds, 1)[1].data.numpy().squeeze()
                acc = (preds == target.numpy()).mean()
                if (i + 1) % 10 == 0:
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f'
                          % (epoch + 1, num_epochs, i + 1, int(len(train_ds) / batch_size), loss.data[0], acc))

                    # save model
            torch.save(self.state_dict(), 'cnn.pth')
            # Cross validation
            # self.LeavOneOutValidation(val_loader)
        torch.save(self.state_dict(), 'cnn2.pth')

    # def LeavOneOutValidation(self, val_loader):
    #     print('Leave one out VALIDATION ...')
    #     model = CNN(img_size=self.img_size, img_ch=self.img_ch, kernel_size=self.kernel_size,
    #                           pool_size=self.pool_size, n_out=self.n_out, padding=self.padding)
    #     model.load_state_dict(torch.load('./cnn.pth'))
    #     val_losses = []
    #     model.eval()
    #     print(val_loader)
    #     eval_loss = 0
    #     eval_acc = 0
    #     for data in val_loader:
    #         img = data['image']
    #         label = data['labels']
    #         #             img, label=data
    #         img = Variable(img, volatile=True)
    #         label = Variable(label, volatile=True)
    #
    #         out = model(img)
    #         loss = model.criterion(out, label)
    #         eval_loss += loss.data[0] * label.size(0)
    #
    #     print('Leave one out VALIDATION Loss: {:.6f}'.format(eval_loss / (len(val_dataset))))
    #     val_losses.append(eval_loss / (len(val_dataset)))
    #     print()

    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i: i + batch_size]