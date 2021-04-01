import torch
import torch.nn as nn
from . import ResNet4SeqAlign as resNet
from . import SeqFeatNet as seqModel
from .EmbeddingLayer import EmbeddingLayer, OuterConcatenate


# This class will generate the observation score, which is the input of CRF
class ObsModel(nn.Module):
    def __init__(self, device, feat1d=20, feat2d=36,
                 layers1d=[1, 1, 1, 1], neurons1d=[20, 30, 40, 50],
                 layers2d=[1, 2, 5, 2], neurons2d=[16, 32, 64, 128],
                 dilation=[1, 1, 1, 1],
                 seqnet="ResNet", embedding='Seq', pairwisenet="ResNet",
                 block="Basic", activation="ReLU", affine=False,
                 track_running_stats=False):
        super(ObsModel, self).__init__()

        # activation
        if activation == "ReLU":
            act = nn.ReLU()
        elif activation == "TANH":
            act = nn.Tanh()
        elif activation == "ELU":
            act = nn.ELU()

        # seqfeat to create two seq feature by 1d-ResNet
        if seqnet == "ResNet":
            self.seqfeatNet = seqModel.seqResNet(
                    featSize=feat1d, obsSize=feat1d, block=seqModel.seqBlock,
                    layers=layers1d, neurons=neurons1d, activation=act,
                    affine=affine, track_running_stats=track_running_stats
                    ).to(device)
        elif seqnet == "LSTM":
            self.seqfeatNet = seqModel.LSTMfeature(
                    featsize=feat1d, obsSize=feat1d).to(device)

        # combine two seq and get the feature of pairwise sequence
        # convert 1d sequence to 2d matrix
        if embedding == 'Seq':
            self.embedding = EmbeddingLayer(n_in=feat1d, n_out=feat1d).to(
                device)
        elif embedding == 'OuterCat':
            self.embedding = OuterConcatenate(n_in=feat1d, n_out=feat1d).to(
                device)

        # get the score from feature,
        # we use resnet / dilated resnet to capture the feature
        if pairwisenet == "ResNet":
            if block == "Basic":
                blo = resNet.BasicBlock
            elif block == "Bottleneck":
                blo = resNet.Bottleneck
            self.resNetmodel = resNet.ResNet(
                    featNum=feat1d+feat2d, block=blo,
                    layers=layers2d, neurons=neurons2d, dilation=dilation,
                    activation=act, affine=affine,
                    track_running_stats=track_running_stats).to(device)

    def forward(self, featdata, seqX, seqY, maskX, maskY):
        # get input ready for the network
        featX, featY = self.seqfeatNet(seqX, seqY, maskX, maskY)
        seqfeat = self.embedding(featX, featY)
        # the feature is combine featdata and seqfeat(generate by PSSM)
        featdata = featdata.to(seqfeat.device)
        featdata = torch.cat([featdata, seqfeat], dim=-1)
        featdata = self.resNetmodel(featdata, maskX, maskY)
        return featdata


# This class will generate the observation score, which is the input of CRF
# This class is design for replace ADMM for NDThreader
class ADMMModel(nn.Module):
    def __init__(self, device, featSize=12,
                 layers2d=[1, 2, 5, 2], neurons2d=[16, 32, 64, 128],
                 dilation=[1, 1, 1, 1], pairwisenet="ResNet",
                 block="Basic", activation="ReLU", affine=False,
                 track_running_stats=False):
        super(ADMMModel, self).__init__()

        if activation == "ReLU":
            act = nn.ReLU()
        elif activation == "TANH":
            act = nn.Tanh()
        elif activation == "ELU":
            act = nn.ELU()

        if pairwisenet == "ResNet":
            if block == "Basic":
                blo = resNet.BasicBlock
            elif block == "Bottleneck":
                blo = resNet.Bottleneck

        self.resNetmodel = resNet.ResNet(
                featNum=featSize+3, block=blo,
                layers=layers2d, neurons=neurons2d, dilation=dilation,
                activation=act, affine=affine,
                track_running_stats=track_running_stats).to(device)

    def forward(self, feature, distanceFeature, maskX, maskY):
        feature = torch.cat([feature, distanceFeature], -1)
        feature = self.resNetmodel(feature, maskX, maskY)
        return feature
