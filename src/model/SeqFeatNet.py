import torch
import torch.nn as nn


# for 1d ResNet
class seqBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU,
                 downsample=None, affine=False, track_running_stats=False):
        super(seqBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7,
                               padding=3, bias=False)
        self.bn1 = nn.InstanceNorm1d(planes, affine=affine,
                                     track_running_stats=track_running_stats)
        self.activation = activation
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7,
                               padding=3, bias=False)
        self.bn2 = nn.InstanceNorm1d(planes, affine=affine,
                                     track_running_stats=track_running_stats)
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


# we must convert the DataSet to a tensor with shape
#                              (len(DataSet), max(xLen), max(yLen), k)
# we must return the new DataSet and a mask matrix to
# deal with the padding problem
# get the feature from pair of seq by 1D resNet
# data must be a list with pair of seq
class seqResNet(nn.Module):
    def __init__(self, featSize, obsSize, block, layers, neurons, activation,
                 affine=False, track_running_stats=False):
        super(seqResNet, self).__init__()
        self.featin = featSize
        self.inplanes = featSize
        self.obsSize = obsSize

        self.conv1 = nn.Conv1d(featSize, 20, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.InstanceNorm1d(20, affine=affine,
                                     track_running_stats=track_running_stats)
        self.activation = activation
        self.layer1 = self._make_layer(
            block, neurons[0], layers[0], activation=activation,
            affine=affine, track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(
            block, neurons[1], layers[1], stride=1, activation=activation,
            affine=affine, track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(
            block, neurons[2], layers[2], stride=1, activation=activation,
            affine=affine, track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(
            block, neurons[3], layers[3], stride=1, activation=activation,
            affine=affine, track_running_stats=track_running_stats)

        self.conv2 = nn.Conv1d(
                neurons[3] * block.expansion, self.obsSize, kernel_size=3,
                stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm1d(
            self.obsSize, affine=affine,
            track_running_stats=track_running_stats)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight)
            elif isinstance(m, nn.InstanceNorm1d):
                continue

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU,
                    affine=False, track_running_stats=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv1d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.InstanceNorm1d(planes * block.expansion, affine=affine,
                                      track_running_stats=track_running_stats)
                    )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride,
                  activation=activation, downsample=downsample, affine=affine,
                  track_running_stats=track_running_stats))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation,
                                affine=affine,
                                track_running_stats=track_running_stats))
        return nn.Sequential(*layers)

    def _make_1d_padding(self, feat, mask):
        for batch in range(len(mask)):
            feat[batch][:][mask[batch]:].zero_()
        return feat

    def generateFeature(self, x, mask):
        # input with shape (batchSize, seqX, feat)
        x = x.permute([0, 2, 1])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self._make_1d_padding(x, mask)
        x = self.bn2(x)

        x = x.permute([0, 2, 1])

        return x

    def forward(self, seqX, seqY, maskX, maskY):
        # return the seq after 1D conv with shape [batchSize, seqX, featsize]
        featX = self.generateFeature(seqX, maskX)
        featY = self.generateFeature(seqY, maskY)

        return featX, featY


class LSTMfeature(nn.Module):
    def __init__(self, featsize, obsSize, hidden_size=8):
        super(LSTMfeature, self).__init__()
        self.lstm = nn.LSTM(input_size=featsize, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.obsSize = obsSize
        self.featsize = featsize
        self.hidden_size = hidden_size
        self.hidden2obs = nn.Linear(self.hidden_size, self.obsSize)

    def forward(self, seqX, seqY, maskX, maskY):
        batchSize = seqX.size(0)
        assert seqY.size(0) == batchSize, "the batchSize should be equal"
        self.hidden = self.init_hidden(batchSize, seqX.device)
        total_length_X = seqX.size(1)
        total_length_Y = seqY.size(1)
        pack_x = torch.nn.utils.rnn.pack_padded_sequence(
                seqX, maskX, batch_first=True, enforce_sorted=False)
        pack_y = torch.nn.utils.rnn.pack_padded_sequence(
                seqY, maskY, batch_first=True, enforce_sorted=False)
        out1, _ = self.lstm(pack_x, self.hidden)
        padded_out1, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out1, batch_first=True, total_length=total_length_X)
        xfeat = self.hidden2obs(padded_out1)

        out2, _ = self.lstm(pack_y, self.hidden)
        padded_out2, _ = torch.nn.utils.rnn.pad_packed_sequence(
                out2, batch_first=True, total_length=total_length_Y)
        yfeat = self.hidden2obs(padded_out2)

        return xfeat, yfeat

    def init_hidden(self, batchSize, device):
        return (torch.randn(1, batchSize, self.hidden_size).to(device),
                torch.randn(1, batchSize, self.hidden_size).to(device))
