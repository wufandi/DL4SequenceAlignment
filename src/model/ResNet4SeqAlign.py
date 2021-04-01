import torch.nn as nn
import math


# for BasicBlock
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


#  for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU,
                 downsample=None, dilation=(1, 1), affine=False,
                 track_running_stats=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.InstanceNorm2d(
            planes, affine=affine, track_running_stats=track_running_stats)
        self.activation = activation
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.InstanceNorm2d(
            planes, affine=affine, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU,
                 downsample=None, dilation=(1, 1), affine=False,
                 track_running_stats=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(
                planes, affine=affine, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5, stride=stride,
                               padding=2*dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.InstanceNorm2d(
            planes, affine=affine, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(
            planes * self.expansion, affine=affine,
            track_running_stats=track_running_stats)
        self.activation = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.activation(out)

        return out


class ResNet(nn.Module):
    def __init__(self, featNum, block, layers, neurons, dilation=[1, 1, 1, 1],
                 numStates=3, activation=nn.ReLU(), affine=False,
                 track_running_stats=False):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.c_in = featNum
        self.c_out = numStates
        self.conv1 = nn.Conv2d(self.c_in, 16, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn1 = nn.InstanceNorm2d(16, affine=affine,
                                     track_running_stats=track_running_stats)
        self.activation = activation
        self.layer1 = self._make_layer(
            block, neurons[0], layers[0], dilation=dilation[0],
            activation=activation, affine=affine,
            track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(
            block, neurons[1], layers[1], stride=1, dilation=dilation[1],
            activation=activation, affine=affine,
            track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(
            block, neurons[2], layers[2], stride=1, dilation=dilation[2],
            activation=activation, affine=affine,
            track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(
            block, neurons[3], layers[3], stride=1, dilation=dilation[3],
            activation=activation, affine=affine,
            track_running_stats=track_running_stats)

        self.conv2 = nn.Conv2d(neurons[3] * block.expansion, self.c_out,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.InstanceNorm2d(
                self.c_out, affine=affine,
                track_running_stats=track_running_stats)

        # to fit the input (batchSize, 3, xLen, yLen) for the CRF algorithm
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            else:
                continue

    def _make_padding(self, feat, maskX, maskY):
        for batch in range(len(maskX)):
            feat[batch, :, maskX[batch]:, :].zero_()
            feat[batch, :, :, maskY[batch]:].zero_()
        return feat

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    activation=nn.ReLU(), affine=False,
                    track_running_stats=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion, affine=affine,
                                  track_running_stats=track_running_stats),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, activation, downsample,
                  affine=affine, track_running_stats=track_running_stats))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation),
                                activation=activation, affine=affine,
                                track_running_stats=track_running_stats))

        return nn.Sequential(*layers)

    # the input have shape with (batchSize, xLen, yLen, feature)
    # while feature stands for channel
    def forward(self, x, maskX, maskY):
        x = x.permute([0, 3, 1, 2])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self._make_padding(x, maskX, maskY)
        x = self.bn2(x)
        # transform shape (batchSize, feature/State, xLen, yLen)
        # to (batch, xLen, yLen, State) to fit the observations
        x = x.permute([0, 2, 3, 1])

        return x
