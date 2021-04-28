import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import math
import matplotlib.pyplot as plt
__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.auxiliary_classifier_layer = nn.Sequential(
            # nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
            # nn.AvgPool2d((5, 5),stride=3),
            # nn.Conv2d(128,3,kernel_size=1)
        )
        self.auxiliary_classifier1 = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        self.auxiliary_classifier2 = nn.Sequential(
            nn.Linear(256, num_classes)
        )
        # self.auxiliary_classifier2 = nn.Sequential(
        #     nn.Linear(256, num_classes)
        # )
        self.use_aux_classify = False
        ##########
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # self.fc = nn.Linear(1280 * 4 * block.expansion, num_classes)#7680

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        global spp
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = math.floor((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            w_pad = math.floor((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid),padding=(h_pad,w_pad))
            x = maxpool(previous_conv)
            if (i == 0):
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def scale_X (self,x):
        global x_cat
        new_X = torch.Tensor(1, 256, 32, 32)
        for i in range(len(x)):
            for j in range(len(x[0])):
                charact_img = x[i][j]
                # charact_img = charact_img.reshape(1,-1)
                charact_img = charact_img.flatten()
                x_1 = charact_img[:1024]
                x_2 = charact_img[1024:2048]
                x_3 = charact_img[2048:3072]
                x_4 = charact_img[3072:4096]
                # y = int(len(x_1) ** 0.5)
                x_1 = x_1.reshape(32, 32)

                x_2 = x_2.reshape(32, 32)
                x_3 = x_3.reshape(32, 32)
                x_4 = x_4.reshape(32, 32)

                x_stack = torch.stack([x_1, x_2, x_3, x_4], dim=0)
                # x_stack = torch.stack([x_1, x_2], dim=0)
                if j == 0:
                    x_cat = x_stack
                else:
                    x_cat = torch.cat((x_cat, x_stack), dim=0)
            new_X[i] = x_cat
        return new_X

    def cutting_2K(self,x):
        num = int(len(x[0][0][0])/2)
        x_1 = x[:,:,0:num,:]
        x_2 = x[:,:,num:,:]
        new_x = torch.cat((x_1,x_2),1)
        return new_x

    def cutting_4K(self,x):
        num = int(len(x[0][0][0])/2)
        x_1 = x[:,:,0:num,0:num]
        x_2 = x[:,:,0:num,num:]
        x_3 = x[:,:,num:,0:num]
        x_4 = x[:,:,num:,num:]
        new_x = torch.cat((x_1,x_2,x_3,x_4),1)
        return new_x

    def cutting_6K(self, x):
        num = int(len(x[0][0][0]) / 2)
        num2 = int(len(x[0][0][0]) / 3)
        x_1 = x[:, :, 0:num, 0:num2]
        x_2 = x[:, :, 0:num, num2:num2*2]
        x_3 = x[:, :, 0:num, num2*2:num2*3]
        x_4 = x[:, :, num:, 0:num2]
        x_5 = x[:, :, num:, num2:num2*2]
        x_6 = x[:, :, num:, num2*2:num2*3]
        new_x = torch.cat((x_1, x_2, x_3, x_4,x_5,x_6), 1)
        return new_x
    def cutting_8K(self, x):
        num1 = int(len(x[0][0][0]) / 2)
        num2 = int(len(x[0][0][0]) / 4)
        x_1 = x[:, :, 0:num1, 0:num2]
        x_2 = x[:, :, 0:num1, num2:2*num2]
        x_3 = x[:, :, 0:num1, 2*num2:3*num2]
        x_4 = x[:, :, 0:num1, 3*num2:]
        x_5 = x[:, :, num1:, 0:num2]
        x_6 = x[:, :, num1:, num2:2 * num2]
        x_7 = x[:, :, num1:, 2 * num2:3 * num2]
        x_8 = x[:, :, num1:, 3 * num2:]
        new_x = torch.cat((x_1, x_2, x_3, x_4,x_5,x_6,x_7,x_8), 1)
        #512*32*16
        return new_x

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        global aux, aux1,aux2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # for i in range(len(x.squeeze())):
        #     plt.matshow(x.squeeze()[i], cmap=plt.cm.gray)
        #     plt.show()
        x = self.layer2(x)
        if self.use_aux_classify:
            tem = self.auxiliary_classifier_layer(x)
            tem = torch.flatten(tem, 1)
            # print(tem.shape)
            aux1 = self.auxiliary_classifier1(tem)

        x = self.layer3(x)
        if self.use_aux_classify:
            tem = self.auxiliary_classifier_layer(x)
            tem = torch.flatten(tem, 1)
            # print(tem.shape)
            aux2 = self.auxiliary_classifier2(tem)
        # x = self.cutting_4K(x)
        x = self.layer4(x)
        # 512*2*2
        x = self.avgpool(x)
        #512*1*1
        # x = self.spatial_pyramid_pool(x, x.shape[0], (x.shape[2],x.shape[3]), (1,2))
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc(x)
        if self.use_aux_classify:
            return x,aux1,aux2
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


