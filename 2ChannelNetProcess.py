import os

import torchvision
from torchvision import datasets, models, transforms

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import constant
import util
from dataset import SiameseNetworkDataset, TwoChannelNetworkDataset
from loss import ContrastiveLoss
from network import SiameseNetwork, PseudoSiameseNetwork
import time
import copy
import dataAnalyze
from torch.optim import lr_scheduler

if not os.path.exists(constant.image_cache_dir):
    os.makedirs(constant.image_cache_dir)


def prepare_train_data_set(data_dir, gray_image=True):
    folder_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
    data_transforms = prepare_data_transforms(gray_image)
    two_channel_datasets = {
        x: TwoChannelNetworkDataset(image_folder_dataset=folder_datasets[x], transform=data_transforms[x])
        for x in ['train', 'val']}
    return two_channel_datasets


def prepare_test_data_set(data_dir, gray_image=True):
    folder_dataset = datasets.ImageFolder(data_dir)
    data_transforms = prepare_data_transforms(gray_image)
    two_channel_datasets = TwoChannelNetworkDataset(image_folder_dataset=folder_dataset, transform=data_transforms["val"])
    return two_channel_datasets


def prepare_data_transforms(gray_image=True):
    resize_value = 512
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(resize_value),
            transforms.RandomCrop(resize_value),
            # transforms.Resize((256, 256)),
            # 随机在图像上裁剪出224*224大小的图像
            # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.Grayscale(1) if gray_image else transforms.Grayscale(3),
            # 将图像随机翻转
            # transforms.RandomHorizontalFlip(),
            # 将图像数据,转换为网络训练所需的tensor向量
            transforms.ToTensor(),
            # 图像归一化处理
            # 个人理解,前面是3个通道的均值,后面是3个通道的方差
            # transforms.Normalize([0.485, ],
            #                      [0.229, ]) if gray_image else transforms.Normalize(
            #     [0.485, 0.456, 0.406],
            #     [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize_value),
            transforms.RandomCrop(resize_value),
            # transforms.Resize((256, 256)),
            # 随机在图像上裁剪出224*224大小的图像
            # transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.Grayscale(1) if gray_image else transforms.Grayscale(3),
            # 将图像随机翻转
            # transforms.RandomHorizontalFlip(),
            # 将图像数据,转换为网络训练所需的tensor向量
            transforms.ToTensor(),
            # 图像归一化处理
            # 个人理解,前面是3个通道的均值,后面是3个通道的方差
            # transforms.Normalize([0.485, ],
            #                      [0.229, ]) if gray_image else transforms.Normalize(
            #     [0.485, 0.456, 0.406],
            #     [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


# 训练与验证网络（所有层都参加训练）
def train_model(model, data_loaders, dataset_sizes, use_gpu, criterion, optimizer, scheduler, num_epochs=25,
                model_name='two_channel_net'):
    since = time.time()
    # 保存网络训练最好的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_history = {x: [] for x in ['train', 'val']}
    counter = {x: [] for x in ['train', 'val']}
    iteration_number = {x: 0 for x in ['train', 'val']}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每训练一个epoch，测试一下网络模型的准确率
        for phase in ['train', 'val']:
            if phase == 'train':
                # 梯度清零
                optimizer.zero_grad()
                # 学习率更新方式
                if epoch > 0:
                    scheduler.step()
                #  调用模型训练
                model.train(True)
            else:
                # 调用模型测试
                model.train(False)
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            softmax = nn.Softmax(dim=1)
            results = []
            signs = []

            # 依次获取所有图像，参与模型训练或测试
            for i, data in enumerate(data_loaders[phase]):
                # 获取输入
                inputs, labels = data
                # 判断是否使用gpu
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # 梯度清零
                optimizer.zero_grad()

                outputs = model(inputs)
                # 计算Loss值
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs.data, 1)

                # 反传梯度，更新权重
                if phase == 'train':
                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()

                # 计算一个epoch的loss值和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if i % 10 == 0:
                    iteration_number[phase] += 10
                    counter[phase].append(iteration_number[phase])
                    loss_history[phase].append(loss.item())
                # if epoch == num_epochs - 1:
                result_np = softmax(outputs).cpu().data.T[1].numpy()

                results = np.concatenate((results, result_np))
                signs = np.concatenate((signs, labels.cpu().data.numpy()))

            precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs.flatten(),
                                                                                      results.flatten())
            if epoch == num_epochs - 1:
                util.save_pr_csv_data(signs, results, precision, recall, threshold, phase)
            # print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)
            print("{} results: {}\nsigns: {}".format(phase, results, signs))
            average_precision = dataAnalyze.compute_average_precision(signs.flatten(),
                                                                      results.flatten())
            print("{} average_precision: {}".format(phase, average_precision))
            dataAnalyze.save_precision_recall_curve_image(precision, recall, average_precision,
                                                          phase + " PR Curve_" + str(epoch),
                                                          "{} PRCure".format(model_name))

            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 保存测试阶段，准确率最高的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            dataAnalyze.save_loss_plot(counter[phase], loss_history[phase], "loss_" + phase + "_" + str(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 网络导入最好的网络权重
    # model.load_state_dict(best_model_wts)
    return model


def img_show(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def generate_resnet_model(use_gpu):
    model = models.resnet18(pretrained=True)
    """
    # 获取最后一个全连接层的输入通道数
    num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    feature_model = list(model.classifier.children())
    # 去掉原来的最后一层
    feature_model.pop()
    # 添加上适用于自己数据集的全连接层
    feature_model.append(nn.Linear(num_input, 2))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层，还可以为网络添加新的层，重新生成网络的后半部分
    model.classifier = nn.Sequential(*feature_model)
    """
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
    print("model: ", model)
    if use_gpu:
        model = model.cuda()
    return model


def two_channel_net_train(data_loaders, dataset_sizes, use_gpu, num_epochs, model_name='two_channel_net'):
    model = generate_resnet_model(use_gpu)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model = train_model(model, data_loaders, dataset_sizes, use_gpu, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs, model_name)
    torch.save(model.state_dict(), constant.cache_dir + "trained_{}.pkl".format(model_name))
    # visualize_model(model)


def prepare_train_data(gray_image, data_dir):
    # 这种数据读取方法,需要有train和val两个文件夹，每个文件夹下一类图像存在一个文件夹下
    image_datasets = prepare_train_data_set(data_dir, gray_image)
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x
                    in
                    ['train', 'val']}
    # 读取数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # 数据类别
    # class_names = image_datasets['train'].classes
    return data_loaders, dataset_sizes


def prepare_test_data(gray_image, data_dir):
    image_dataset = prepare_test_data_set(data_dir, gray_image)
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True, num_workers=4)
    return data_loader, image_dataset


def net_test(data_loader, dataset, use_gpu, model_name="two_channel_net"):
    model = generate_resnet_model(use_gpu)

    softmax = nn.Softmax(dim=1)
    results = []
    signs = []
    map_location = torch.device('cpu')
    if use_gpu:
        model = model.cuda()
        map_location = None
    model.load_state_dict(
        torch.load(constant.cache_dir + 'trained_{}.pkl'.format(model_name), map_location=map_location))
    model.eval()

    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print("loop {} in {}".format(i, int(len(dataset) / data_loader.batch_size)))
        # 获取输入
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():

            outputs = model(inputs)
            results = np.concatenate((results, softmax(outputs).cpu().data.T[1].numpy()))
        signs = np.concatenate((signs, labels.cpu().data.numpy()))
    precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs.flatten(),
                                                                              results.flatten())
    print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)
    util.save_pr_csv_data(signs, results, precision, recall, threshold, "test")
    print("results: {}\nsigns: {}".format(results, signs))
    average_precision = dataAnalyze.compute_average_precision(signs.flatten(),
                                                              results.flatten())
    print("average_precision: {}".format(average_precision))
    dataAnalyze.show_precision_recall_curve_image(precision, recall, average_precision, "Test PR Curve")


# def test(test_dir_path, net, show_process_image, loop_time_limit=-1, gray_image=True, model_name="two_channel_net"):
#     map_location = torch.device('cpu')
#     signs = []
#     results = []
#     if use_gpu:
#         net = net.cuda()
#         map_location = None
#     net.load_state_dict(torch.load(constant.cache_dir + 'trained_{}.pkl'.format(model_name), map_location=map_location))
#     net.eval()
#
#     data_transforms = prepare_data_transform(gray_image)
#     test_dataset = TwoChannelNetworkDataset(
#         image_folder_dataset=datasets.ImageFolder(test_dir_path),
#         transform=data_transforms["val"])
#     test_data_loader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
#     dataset_size = len(test_dataset)
#     if loop_time_limit < 0:
#         loop_time_limit = dataset_size
#     # test_data_loader = DataLoader(siamese_datasets["val"], num_workers=6, batch_size=1, shuffle=True)
#     for i, data in enumerate(test_data_loader):
#         if i % 10 == 0:
#             print("loop:{} in {}".format(i, int(loop_time_limit / test_data_loader.batch_size)))
#         img0, img1, label = data
#         concatenated = torch.cat((img0, img1), 0)
#
#         x0 = Variable(img0)
#         x1 = Variable(img1)
#         if use_gpu:
#             x0 = x0.cuda()
#             x1 = x1.cuda()
#         with torch.no_grad():
#             output1, output2 = net(x0, x1)
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         sign = int(label.item())
#         result = euclidean_distance.item()
#         if show_process_image:
#             img_show(torchvision.utils.make_grid(concatenated),
#                      'label:{}   dissimilarity: {:.2f}'.format(sign, result))
#         signs.append(sign)
#         results.append(round(result, 3))
#         if i > loop_time_limit:
#             break
#     print("signs:{}\nresults:{}".format(signs, results))
#     precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs, results)
#     print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)
#
#     average_precision = dataAnalyze.compute_average_precision(signs, results)
#     dataAnalyze.show_precision_recall_curve_image(precision, recall, average_precision, is_save=False)


def train(data_dir, num_epochs=50):
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    # 是否灰度图
    gray_image = True

    data_loaders, dataset_sizes = prepare_train_data(gray_image, data_dir)
    # googLeNet_train()
    # alexNet_train()
    two_channel_net_train(data_loaders, dataset_sizes, use_gpu, num_epochs)


def test(data_dir):
    is_image_gray = True
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    data_loader, image_dataset = prepare_test_data(is_image_gray, data_dir)
    net_test(data_loader, image_dataset, use_gpu)


if __name__ == '__main__':
    train(r"D:\Work\dataset\LoveLetterMaterial\SiameseUse_CenterCropLoveLetterIFPSCopy_rotate_crop", 20)
    # test(r"D:\Work\dataset\copyday\copydays_origin_crop")
# train_siamese_net()
# train_pseudo_siamese_net()
