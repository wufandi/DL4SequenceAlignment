import torch
import torch.nn as nn
import numpy as np


class EmbeddingLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(EmbeddingLayer, self).__init__()
        # input is a profile derived from 1d convolution
        # n_in is the number of features in input
        self.n_in = n_in
        self.n_out = n_out

        value_bound = np.sqrt(6./(n_in * n_in + n_out))
        W_values = np.asarray(np.random.uniform(low=-value_bound,
                              high=value_bound, size=(n_in, n_in, n_out)))
        self.W = nn.Parameter(torch.from_numpy(W_values).float())
        self.params = [self.W]
        self.paramL1 = abs(self.W).sum()
        self.paramL2 = (self.W**2).sum()
        self.pcenters = (torch.mean(torch.mean(self.W, 0), 1)).sum()

    def forward(self, input1, input2):
        # input1 shall have shape (batchSize, LenX, n_in)
        # input2 shall have shape (batchSize, LenY, n_in)
        assert input1.size(0) == input2.size(0), "input1 and \
            input2 should have equal batchSize"
        assert input1.size(2) == input2.size(2), "input1 and \
            input2 should have equal n_in"

        # input1:
        #   1. shape to (batchSize, LenX, n_in, n_out)
        #   2. shape to (batchSize, n_out, LenX, n_in)
        input1 = torch.einsum("abc,ccd->abcd", (input1, self.W))
        input1 = input1.permute([0, 3, 1, 2])

        # input2:
        #   1. shape to (batchSize, n_in, LenY)
        input2 = input2.permute([0, 2, 1])

        # out:
        #   1. shape to (batchSize, n_out, LenX, LenY)
        #   2. shape to (batchSize, LenX, LenY, n_out)
        out = torch.einsum("boxi,biy->boxy", (input1, input2))
        out = out.permute([0, 2, 3, 1])

        return out


class OuterConcatenate(nn.Module):
    def __init__(self, n_in, n_out):
        super(OuterConcatenate, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.conv = nn.Conv2d(2*n_in, n_out, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn = nn.InstanceNorm2d(n_out)

    def forward(self, input1, input2):
        # input1 shall have shape (batchSize, LenX, n_in)
        # input2 shall have shape (batchSize, LenY, n_in)
        # output1 shall have shape (batchSize, LenX, LenY, 2*n_in)
        # using an convolution layer,
        # output shall have shape (batchSize, LenX, LenY, n_out)
        assert input1.size(0) == input2.size(0), "input1 and \
            input2 should have equal batchSize"
        assert input1.size(2) == input2.size(2), "input1 and \
            input2 should have equal n_in"
        grid_x, grid_y = torch.meshgrid(
            torch.LongTensor(range(input1.size(1))),
            torch.LongTensor(range(input2.size(1))))
        input1 = input1.permute(1, 0, 2)[grid_x]
        input2 = input2.permute(1, 0, 2)[grid_y]

        out = torch.cat((input1, input2), 3)
        out = out.permute([2, 3, 0, 1])
        out = self.conv(out)
        out = self.bn(out)
        out = out.permute([0, 2, 3, 1])

        return out
