import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from cnn import CNN
from main import train_loader, batch_size

# # Hyper Parameters
# num_epochs = 1
# batch_size = 100
# learning_rate = 0.001
#
# cnn = CNN()
#
# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
#
# optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate) #ADAM OPTIMIZER
#
# # Train the Model
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = Variable(images)
#         labels = Variable(labels)
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = cnn(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print('Epoch [%d/%d], Iter [%d] Loss: %.4f'
#                   % (epoch + 1, num_epochs, i + 1, loss.data[0]))
#
# # Test the Model
# cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images)
#     outputs = cnn(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()
#
# print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

img_size = (75,75)
img_ch = 2
kernel_size = 7
pool_size = 2
padding=2
n_out = 1
n_epoch = 35

if __name__ == '__main__':
    cnn = CNN(img_size=img_size, img_ch=img_ch, kernel_size=kernel_size,
                        pool_size=pool_size, n_out=n_out, padding=padding)
    cnn.fit(train_loader, n_epoch, batch_size)