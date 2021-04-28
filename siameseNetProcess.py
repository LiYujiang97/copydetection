import os

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
from dataset import SiameseNetworkDataset
from loss import ContrastiveLoss
from network import SiameseNetwork, PseudoSiameseNetwork
import time
import copy
import dataAnalyze
from torch.optim import lr_scheduler

if not os.path.exists(constant.image_cache_dir):
    os.makedirs(constant.image_cache_dir)

# 是否使用gpu运算
use_gpu = torch.cuda.is_available()


def prepare_data_set(data_dir, gray_image=True):
    folder_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['train', 'val']}
    data_transforms = prepare_data_transform(gray_image)
    siamese_datasets = {
        x: SiameseNetworkDataset(image_folder_dataset=folder_datasets[x], transform=data_transforms[x])
        for x in ['train', 'val']}
    return siamese_datasets


def prepare_data_transform(gray_image=True):
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


def train_model(siamese_datasets, net, optimizer, scheduler, criterion, epochs_num=50, model_name='siamese_net'):
    # 读取数据集大小
    dataset_sizes = {x: len(siamese_datasets[x]) for x in ['train', 'val']}
    since = time.time()
    if use_gpu:
        net = net.cuda()

    data_loaders = {x: DataLoader(siamese_datasets[x], batch_size=16, shuffle=True, num_workers=8) for x
                    in
                    ['train', 'val']}
    best_model_wts = None
    counter = []
    loss_history = []
    iteration_number = 0
    best_loss = 10000
    for epoch in range(0, epochs_num):
        # 每训练一个epoch，测试一下网络模型的准确率
        for phase in ['train', 'val']:
            if phase == 'train':
                # 梯度清零
                optimizer.zero_grad()
                if epoch > 0:
                    # 学习率更新方式
                    scheduler.step()
                #  调用模型训练
                net.train()
            else:
                # 调用模型测试
                net.train()

            running_loss = 0.0
            running_corrects = 0

            results = []
            signs = []

            for i, data in enumerate(data_loaders[phase], 0):
                img0, img1, label = data
                # print("img0.shape = {},img1.shape = {},label.shape = {}".format(img0.shape, img1.shape, label.shape))
                if use_gpu:
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                # 反传梯度，更新权重
                if phase == 'train':
                    # 反传梯度
                    loss_contrastive.backward()
                    # 更新权重
                    optimizer.step()
                    # print("output1 = {},output2 = {},label = {}".format(output1, output2, label))
                    if i % 10 == 0:
                        print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                        iteration_number += 10
                        counter.append(iteration_number)
                        loss_history.append(loss_contrastive.item())

                running_loss += loss_contrastive.item() * label.size(0)

                pairwise_distance = F.pairwise_distance(output1, output2, keepdim=True)
                distance_numpy = pairwise_distance.cpu().data.numpy().T
                results = np.concatenate((results, distance_numpy.reshape(-1)))
                label_numpy = label.cpu().data.numpy().T
                signs = np.concatenate((signs, label_numpy.reshape(-1)))

                if use_gpu:
                    torch.cuda.empty_cache()

            # precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs.flatten(),results.flatten())
            precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs, results)

            if epoch == epochs_num - 1:
                util.save_pr_csv_data(signs, results, precision, recall, threshold, phase)
            # print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)

            # average_precision = dataAnalyze.compute_average_precision(signs.flatten(), results.flatten())
            average_precision = dataAnalyze.compute_average_precision(signs, results)
            dataAnalyze.save_precision_recall_curve_image(precision, recall, average_precision,
                                                          phase + " PR Curve_" + str(epoch),
                                                          "{}_PRCurve".format(model_name))
            if epoch % 10 == 0 or epoch == epochs_num - 1:
                print("{} results: {}\nsigns: {}".format(phase, results, signs))
                print("{} average_precision: {}".format(phase, average_precision))

            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = float(running_corrects) / dataset_sizes[phase]
            #
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 保存测试阶段，損失最低的模型
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
        dataAnalyze.save_loss_plot(counter, loss_history, "loss_" + phase + "_" + str(epoch),
                                   "{}LossCurve".format(model_name))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    # 网络导入最好的网络权重
    # net.load_state_dict(best_model_wts)
    return net


def test(test_dir_path, net, show_process_image, loop_time_limit=-1, gray_image=True, model_name="siamese_net"):
    map_location = torch.device('cpu')
    signs = []
    results = []
    if use_gpu:
        net = net.cuda()
        map_location = None
    net.load_state_dict(torch.load(constant.cache_dir + 'trained_{}.pkl'.format(model_name), map_location=map_location))
    net.eval()

    data_transforms = prepare_data_transform(gray_image)
    test_dataset = SiameseNetworkDataset(
        image_folder_dataset=datasets.ImageFolder(test_dir_path),
        transform=data_transforms["val"])
    test_data_loader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
    dataset_size = len(test_dataset)
    if loop_time_limit < 0:
        loop_time_limit = dataset_size
    # test_data_loader = DataLoader(siamese_datasets["val"], num_workers=6, batch_size=1, shuffle=True)
    for i, data in enumerate(test_data_loader):
        if i % 10 == 0:
            print("loop:{} in {}".format(i, int(loop_time_limit / test_data_loader.batch_size)))
        img0, img1, label = data
        concatenated = torch.cat((img0, img1), 0)

        x0 = Variable(img0)
        x1 = Variable(img1)
        if use_gpu:
            x0 = x0.cuda()
            x1 = x1.cuda()
        with torch.no_grad():
            output1, output2 = net(x0, x1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        sign = int(label.item())
        result = euclidean_distance.item()
        if show_process_image:
            img_show(torchvision.utils.make_grid(concatenated),
                     'label:{}   dissimilarity: {:.2f}'.format(sign, result))
        signs.append(sign)
        results.append(round(result, 3))
        if i > loop_time_limit:
            break
    print("signs:{}\nresults:{}".format(signs, results))
    precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs, results)
    print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)

    average_precision = dataAnalyze.compute_average_precision(signs, results)
    dataAnalyze.show_precision_recall_curve_image(precision, recall, average_precision, is_save=False)


def img_show(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(siamese_net, epochs_num=20, model_name='siamese_net'):
    optimizer_ft = torch.optim.SGD(siamese_net.parameters(), lr=0.001, momentum=0.9)
    scheduler_lr = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    criterion = ContrastiveLoss()

    siamese_datasets = prepare_data_set(
        r"D:\Work\dataset\LoveLetterMaterial\SiameseUse_CenterCropLoveLetterIFPSCopy_rotate_crop")
    model = train_model(siamese_datasets, siamese_net, optimizer_ft, scheduler_lr, criterion,
                        epochs_num, model_name)
    torch.save(model.state_dict(), constant.cache_dir + "trained_{}.pkl".format(model_name))


def prepare_siamese_net_model(use_customer_net_imp, gray_image=True):
    customer_net_imp = generate_customer_net_imp(gray_image, use_customer_net_imp)
    return SiameseNetwork(customer_net_imp)


def prepare_pseudo_siamese_net_model(use_customer_net_imp, gray_image=True):
    customer_net_imp = generate_customer_net_imp(gray_image, use_customer_net_imp)
    return PseudoSiameseNetwork(customer_net_imp)


def generate_customer_net_imp(gray_image, use_customer_net_imp):
    customer_net_imp = None
    if use_customer_net_imp:
        resnet18 = torchvision.models.resnet18(pretrained=True)
        if gray_image:
            # 灰度图输入，所以第一层的input维度改为1
            resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, 5)
        customer_net_imp = resnet18
    return customer_net_imp


def train_pseudo_siamese_net():
    net_model = prepare_pseudo_siamese_net_model(True)
    train(net_model, 20, "pseudo_siamese_net")


def train_siamese_net():
    net_model = prepare_siamese_net_model(True)
    train(net_model, 20, "siamese_net")


def test_pseudo_siamese_net(dir_path):
    net_model = prepare_pseudo_siamese_net_model(True)
    test(dir_path, net_model, False, -1, model_name="pseudo_siamese_net")


def test_siamese_net(dir_path):
    net_model = prepare_siamese_net_model(True)
    test(dir_path, net_model, False, -1, model_name="siamese_net")


if __name__ == '__main__':
    # train_siamese_net()
    # train_pseudo_siamese_net()
    test_siamese_net(r"D:\Work\dataset\copyday\copydays_origin_crop")
    # test_pseudo_siamese_net(r"D:\Work\dataset\copyday\copydays_original_jpeg")
