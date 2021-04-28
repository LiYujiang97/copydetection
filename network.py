from copy import deepcopy

import torch.nn as nn


# 孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self, net_implement=None):
        super(SiameseNetwork, self).__init__()

        self.net_implement = net_implement
        if net_implement is None:
            self.cnn1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, 4, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(4),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.ReflectionPad2d(1),
                nn.Conv2d(4, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),

                nn.ReflectionPad2d(1),
                nn.Conv2d(8, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),

            )

            self.fc1 = nn.Sequential(
                nn.Linear(524288, 500),
                # nn.Linear(8 * 100 * 100, 500),
                nn.ReLU(inplace=True),

                nn.Linear(500, 500),
                nn.ReLU(inplace=True),

                nn.Linear(500, 5))

    def forward_once(self, x):
        if self.net_implement is None:
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
        else:
            output = self.net_implement(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 伪孪生网络
class PseudoSiameseNetwork(nn.Module):
    def __init__(self, net_implement=None):
        super(PseudoSiameseNetwork, self).__init__()

        self.net_implement1 = net_implement
        self.net_implement2 = net_implement
        if net_implement is None:
            nn_sequential = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, 4, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(4),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                nn.ReflectionPad2d(1),
                nn.Conv2d(4, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),

                nn.ReflectionPad2d(1),
                nn.Conv2d(8, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),

            )
            self.cnn1 = nn_sequential
            self.cnn2 = deepcopy(nn_sequential)

            fc_sequential = nn.Sequential(
                nn.Linear(524288, 500),
                # nn.Linear(8 * 100 * 100, 500),
                nn.ReLU(inplace=True),

                nn.Linear(500, 500),
                nn.ReLU(inplace=True),

                nn.Linear(500, 5))
            self.fc1 = fc_sequential
            self.fc2 = deepcopy(fc_sequential)

    def forward_1(self, x):
        if self.net_implement1 is None:
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
        else:
            output = self.net_implement1(x)
        return output

    def forward_2(self, x):
        if self.net_implement2 is None:
            output = self.cnn2(x)
            output = output.view(output.size()[0], -1)
            output = self.fc2(output)
        else:
            output = self.net_implement2(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_1(input1)
        output2 = self.forward_2(input2)
        return output1, output2
