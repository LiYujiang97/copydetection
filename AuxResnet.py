import torchvision.models.resnet as resnet
import torch.nn as nn
import torch
from torchvision.models.utils import load_state_dict_from_url


class AuxResnet(resnet.ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block, layers, num_classes, zero_init_residual,
                         groups, width_per_group, replace_stride_with_dilation,
                         norm_layer)
        # self.auxiliaryClassifier = AuxiliaryClassifier(128, num_classes)
        self.auxiliary_classifier_layer = nn.Sequential(
            # nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.auxiliary_classifier1 = nn.Sequential(
            nn.Linear(64, num_classes)
        )
        self.auxiliary_classifier2 = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        self.use_aux_classify = False

    def _forward_impl(self, x):
        aux = None
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.use_aux_classify:
            tem = self.auxiliary_classifier_layer(x)
            tem = torch.flatten(tem, 1)
            aux = self.auxiliary_classifier1(tem)

        x = self.layer2(x)

        print("x.shape = {},x.size() = {},x.ndim = {}".format(x.shape, x.size(), x.ndim))
        # if self.training and self.use_aux_classify:
        if self.use_aux_classify:
            tem = self.auxiliary_classifier_layer(x)
            tem = torch.flatten(tem, 1)
            aux = (aux, self.auxiliary_classifier2(tem))

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # if self.training and self.use_aux_classify:
        if self.use_aux_classify:
            return x, aux
        return x


# class AuxiliaryClassifier(nn.Module):
#
#     def __init__(self, in_channels, num_classes):
#         super(AuxiliaryClassifier, self).__init__()
#         # self.conv0 = nn.Conv2d(in_channels, 128, kernel_size=1)
#         # self.conv1 = nn.Conv2d(128, 768, kernel_size=5)
#         # self.conv1.stddev = 0.01
#         # self.fc = nn.Linear(768, num_classes)
#         self.fc = nn.Linear(in_channels, num_classes)
#         # self.fc.stddev = 0.001
#
#     def forward(self, x):
#         # 17 x 17 x 768
#         # x = F.avg_pool2d(x, kernel_size=5, stride=3)
#         # 5 x 5 x 768
#         # x = self.conv0(x)
#         # 5 x 5 x 128
#         # x = self.conv1(x)
#         # 1 x 1 x 768
#         x = x.view(x.size(0), -1)
#         # 768
#         x = self.fc(x)
#         # 1000
#         return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = AuxResnet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(resnet.model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', resnet.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)
