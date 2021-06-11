import torch
import torch.nn as nn


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=64),
            nn.ReLU()
        )

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=128),
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=256),
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=512),
            nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels=512),
            nn.ReLU()
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc0 = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.BatchNorm2d(4096),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.BatchNorm2d(4096),
            nn.ReLU()
        )

        self.final = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv0(x)
        x = self.pool0(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool4(x)
        x = self.conv4(x)
        net_features = self.pool4(x)

        x = net_features.view(x.size(0), -1)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.final(x)

        return net_features, x
