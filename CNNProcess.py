import util
from googLeNet import googLeNet_official
import torch
from PIL import Image
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
import constant
import resnet
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

import dataAnalyze
import cv2
import AuxResnet


def resNet_valuate(data_loader, dataset, use_gpu, is_image_gray=True, model_name="resNet"):
    model = generate_resnet_model(is_image_gray, use_gpu)

    map_location = torch.device('cpu')
    if use_gpu:
        map_location = None
    model.load_state_dict(
        torch.load(constant.cache_dir + 'trained_{}.pkl'.format(model_name), map_location=map_location))
    model.eval()

    valuate_model(data_loader, dataset, model, use_gpu)


def valuate_model(data_loader, dataset, model, use_gpu):
    since = time.time()
    # 数据类别
    class_names = dataset.classes
    softmax = nn.Softmax(dim=1)
    results = []
    signs = []
    aux_weight1 = 0.2
    aux_weight2 = 0.2

    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print("loop {} in {}".format(i, int(len(dataset) / data_loader.batch_size)))
        # 获取输入
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            if model.use_aux_classify:
                outputs, aux_output1 ,aux_output2= model(inputs)
                # aux_output1, aux_output2 = aux_outputs
                # print("valuate outputs = {}\naux_outputs = {}".format(outputs, aux_outputs))
                results = np.concatenate((results, (1 - aux_weight1 - aux_weight2) *
                                          softmax(outputs).cpu().data.T[1].numpy() +
                                          aux_weight1 * softmax(aux_output1).cpu().data.T[1].numpy() +
                                          aux_weight2 * softmax(aux_output2).cpu().data.T[1].numpy()))
            else:
                outputs = model(inputs)
                results = np.concatenate((results, softmax(outputs).cpu().data.T[1].numpy()))
        signs = np.concatenate((signs, labels.cpu().data.numpy()))
    precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs.flatten(),
                                                                              results.flatten())
    # print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)
    # util.save_pr_csv_data(signs, results, precision, recall, threshold, "test")
    # print("results: {}\nsigns: {}".format(results, signs))
    average_precision = dataAnalyze.compute_average_precision(signs.flatten(),
                                                              results.flatten())
    print("average_precision: {}".format(average_precision))
    time_elapsed = time.time() - since
    print('avgtime',time_elapsed/500)
    dataAnalyze.show_precision_recall_curve_image(precision, recall, average_precision, "Test PR Curve")


# 训练与验证网络（所有层都参加训练）
def train_model(model, data_loaders, dataset_sizes, use_gpu, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 保存网络训练最好的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_history = {x: [] for x in ['train', 'val']}
    counter = {x: [] for x in ['train', 'val']}
    iteration_number = {x: 0 for x in ['train', 'val']}

    aux_weight1 = 0.2
    aux_weight2 = 0.2
    aux_outputs = None
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

                # 网络前向运行
                if model.use_aux_classify:
                    outputs, aux_output1, aux_output2= model(inputs)
                    # aux_output1, aux_output2 = aux_outputs
                    # 计算Loss值
                    loss = criterion(outputs, labels)
                    aux_loss1 = criterion(aux_output1, labels)
                    aux_loss2 = criterion(aux_output2, labels)
                    loss = (1 - aux_weight1 - aux_weight2) * loss + aux_weight1 * aux_loss1 + aux_weight2 * aux_loss2
                else:
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
                if model.use_aux_classify:
                    # aux_output1, aux_output2 = aux_outputs
                    results = np.concatenate((results, (1 - aux_weight1 - aux_weight2) * result_np +
                                              aux_weight1 * softmax(aux_output1).cpu().data.T[1].numpy() +
                                              aux_weight2 * softmax(aux_output2).cpu().data.T[1].numpy()))
                else:
                    results = np.concatenate((results, result_np))
                signs = np.concatenate((signs, labels.cpu().data.numpy()))

            precision, recall, threshold = dataAnalyze.compute_precision_recall_curve(signs.flatten(),
                                                                                      results.flatten())
            if epoch == num_epochs - 1:
                util.save_pr_csv_data(signs, results, precision, recall, threshold, phase)
            # print("precision:", precision, "\nrecall:", recall, "\nthreshold:", threshold)
            # print("{} results: {}\nsigns: {}".format(phase, results, signs))
            average_precision = dataAnalyze.compute_average_precision(signs.flatten(),
                                                                      results.flatten())
            print("{} average_precision: {}".format(phase, average_precision))
            dataAnalyze.save_precision_recall_curve_image(precision, recall, average_precision,
                                                          phase + " PR Curve_" + str(epoch))

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


def alexNet_train(use_gpu):
    # 导入Pytorch封装的AlexNet网络模型
    model = models.alexnet(pretrained=True)
    # 获取最后一个全连接层的输入通道数
    num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    feature_model = list(model.classifier.children())
    feature_model.pop()
    # 添加上适用于自己数据集的全连接层
    feature_model.append(nn.Linear(num_input, 2))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
    # 还可以为网络添加新的层
    # 重新生成网络的后半部分
    model.classifier = nn.Sequential(*feature_model)
    if use_gpu:
        model = model.cuda()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 为不同层设定不同的学习率
    fc_params = list(map(id, model.classifier[6].parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    params = [{"params": base_params, "lr": 0.0001},
              {"params": model.classifier[6].parameters(), "lr": 0.001}, ]
    optimizer_ft = torch.optim.SGD(params, lr=1e-4)
    # 定义学习率的更新方式，每5个epoch修改一次学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    torch.save(model.state_dict(), constant.cache_dir + "model_AlexNet.pkl")
    # visualize_model(model)


def googLeNet_train(use_gpu):
    model = googLeNet_official.googlenet(pretrained=False, init_weights=True, num_classes=17)

    if use_gpu:
        model = model.cuda()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # # 为不同层设定不同的学习率
    # fc_params = list(map(id, model.classifier[6].parameters()))
    # base_params = filter(lambda p: id(p) not in fc_params, model.parameters())
    # params = [{"params": base_params, "lr": 0.0001},
    #           {"params": model.classifier[6].parameters(), "lr": 0.001}, ]
    params = model.parameters()
    optimizer_ft = torch.optim.SGD(params, lr=1e-4)

    # 定义学习率的更新方式，每5个epoch修改一次学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    torch.save(model.state_dict(), constant.cache_dir + "model_googLeNet.pkl")


def resNet_train(data_loaders, dataset_sizes, gray_image, use_gpu, num_epochs=50, model_name="resNet"):
    model = generate_resnet_model(gray_image, use_gpu)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)
    model = train_model(model, data_loaders, dataset_sizes, use_gpu, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs)
    torch.save(model.state_dict(), constant.cache_dir + "trained_{}.pkl".format(model_name))
    # visualize_model(model)


def generate_resnet_model(gray_image, use_gpu):
    model = models.resnet18(pretrained=True)
    # model = AuxResnet.resnet18(pretrained=True)
    # model = resnet.resnet18(pretrained=False)
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
    model.fc = nn.Linear(num_ftrs, 17)
    if gray_image:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    edit_resnet(gray_image, model)
    # print("model: ", model)
    if use_gpu:
        model = model.cuda()
    return model


def edit_resnet(gray_image, model):
    if gray_image:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # del model.maxpool
    # num_in = model.maxpool.in_features
    # print("num_in = ", num_in)
    # model.maxpool = nn.Conv2d(64, 64, kernel_size=1, bias=False)


def prepare_train_data(gray_image, data_dir):
    data_transforms = get_data_transforms(gray_image)
    # 这种数据读取方法,需要有train和val两个文件夹，每个文件夹下一类图像存在一个文件夹下
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=15, shuffle=True, num_workers=4) for x
                    in
                    ['train', 'val']}
    # 读取数据集大小
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # 数据类别
    class_names = image_datasets['train'].classes
    return data_loaders, dataset_sizes


def prepare_valuate_data(gray_image, data_dir):
    data_transforms = get_data_transforms(gray_image)
    image_dataset = datasets.ImageFolder(data_dir, data_transforms["val"])
    data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=15, shuffle=True, num_workers=4)
    return data_loader, image_dataset


def get_data_transforms(gray_image):
    resize_value = 128
    # 数据预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(resize_value),
            # transforms.Resize((32,32)),
            # 随机在图像上裁剪出224*224大小的图像
            # transforms.RandomResizedCrop(224),
            transforms.RandomCrop(resize_value),
            transforms.Grayscale(1) if gray_image else transforms.Grayscale(3),
            # 将图像随机翻转
            # transforms.RandomHorizontalFlip(),
            # 将图像数据,转换为网络训练所需的tensor向量
            transforms.ToTensor(),
            # 图像归一化处理
            # 个人理解,前面是3个通道的均值,后面是3个通道的方差
            # transforms.Normalize([0.485, ], [0.229, ]) if gray_image else transforms.Normalize([0.485, 0.456, 0.406],
            #                                                                                    [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize_value),
            # transforms.CenterCrop(224),
            transforms.RandomCrop(resize_value),
            transforms.Grayscale(1) if gray_image else transforms.Grayscale(3),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, ], [0.229, ]) if gray_image else transforms.Normalize([0.485, 0.456, 0.406],
            #                                                                                    [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def train(data_dir, num_epochs):
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    # 是否灰度图
    gray_image = True
    data_loaders, dataset_sizes = prepare_train_data(gray_image, data_dir)
    resNet_train(data_loaders, dataset_sizes, gray_image, use_gpu, num_epochs, "flowers17")


def valuate(data_dir, is_image_gray):
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    data_loader, image_dataset = prepare_valuate_data(is_image_gray, data_dir)
    resNet_valuate(data_loader, image_dataset, use_gpu, is_image_gray, "flowers17")


if __name__ == '__main__':

    train(r"E:\img_data\Resnxt\cnn_flowers17", 15)
    # valuate(r"D:\img_data\CNN2\test",True)

