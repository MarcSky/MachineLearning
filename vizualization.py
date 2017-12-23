# import torch
# import torch.nn as nn
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# from torch.autograd import Variable
#
# from cnn import CNN
# from matplotlib import pyplot as plt
# import torchvision
#
# # Hyper Parameters
# batch_size = 25
#
# # MNIST Dataset
# test_dataset = dsets.MNIST(root='./data/',
#                            train=False,
#                            transform=transforms.ToTensor())
#
# # Data Loader (Input Pipeline)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
#
#
# def visualize_model(model, num_images):
#     images_so_far = 0
#     fig = plt.figure()
#
#     for i, data in enumerate(test_loader):
#         images, labels = data
#
#         inputs = Variable(images)
#         labels = Variable(labels)
#
#         outputs = model(inputs)
#         _, preds = torch.max(outputs.data, 1)
#
#         for j in range(num_images):
#             ax = plt.subplot(num_images//5, 5, j+1)
#             ax.axis('off')
#             ax.set_title('predicted: {}'.format(str(preds[j])))
#             img = inputs.data[j,0].numpy()
#             ax.imshow(img)
#
#         plt.show()
#
# cnn = CNN()
# cnn.load_state_dict(torch.load('cnn.pth'))
# cnn.eval()
#
# visualize_model(cnn, 25)
