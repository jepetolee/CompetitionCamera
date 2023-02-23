import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from MOE import MoE



class NodeConvolution(Module):

    def __init__(self, kernel, input_size, pooling_size=2):
        super(NodeConvolution, self).__init__()
        self.pooling_size = pooling_size
        self.weight1 = Parameter(torch.FloatTensor(kernel, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size = input.shape[0]
        steps = batch_size // self.pooling_size
        left_tensor = False
        if batch_size % self.pooling_size != 0:
            steps += 1
            left_tensor = True

        tensor_list = list()
        for j in range(steps):
            if left_tensor is True and j == steps - 1:
                tensor=torch.zeros(self.pooling_size,input.shape[1]).cuda()
                for i in range(batch_size % self.pooling_size):
                    tensor[i]=input[self.pooling_size * j + i]
                tensor_list.append(torch.sum(tensor * self.weight1,dim=0))
            else:
                tensor_list.append(torch.sum(input[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size] * self.weight1,dim=0))


        return torch.stack(tensor_list, dim=0)


class WisePooling(Module):

    def __init__(self):
        super(WisePooling, self).__init__()

    def forward(self, input, graph):
        tensor_list =list()
        for j in range(graph.shape[0]):
            shot_boundary = graph[j]
            tensor_list.append(torch.div(torch.sum(input[shot_boundary[0]:shot_boundary[1] + 1], dim=0),shot_boundary[1] - shot_boundary[0] + 1)+6e-3)
        return torch.stack(tensor_list, dim=0)


class WiseConvolution(Module):

    def __init__(self, input_size, output_size):
        super(WiseConvolution, self).__init__()
        self.WiseConv = nn.Linear(input_size, output_size)

    def forward(self, input, graph):
        tensor_list = list()
        for j in range(graph.shape[0]):
            shot_boundary = graph[j]

            tensor_list.append(torch.sum(self.WiseConv(input[shot_boundary[0]:shot_boundary[1] + 1]), dim=0))

        return torch.stack(tensor_list, dim=0)


class GraphAttentionPooling(Module):

    def __init__(self, in_features, pooling_size=3):
        super(GraphAttentionPooling, self).__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, 1, bias=True)
        self.pooling_size = pooling_size

    def forward(self, batch_rep):
        batch_size = batch_rep.shape[0]
        steps = batch_size // self.pooling_size
        left_tensor = False
        if batch_size % self.pooling_size != 0:
            steps += 1
            left_tensor = True

        tensor_list = list()
        for j in range(steps):
            if left_tensor is True and j == steps - 1:
                left_size = self.pooling_size - batch_size % self.pooling_size

                att_w = F.softmax(self.W(
                    batch_rep[self.pooling_size * j + 0 - left_size: self.pooling_size * j + self.pooling_size - left_size]),dim=0)
                tensor_list.append(batch_rep[self.pooling_size * j + 0 - left_size: self.pooling_size * j + self.pooling_size - left_size].T @ att_w)
            else:
                att_w = F.softmax(
                    self.W(batch_rep[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size]), dim=0)
                tensor_list.append(batch_rep[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size].T @ att_w)

        return torch.stack(tensor_list, dim=0)


class DCGN(nn.Module):
    def __init__(self, input, nclass, pooling_size=3):
        super(DCGN, self).__init__()

        self.nodewiseconvolution = NodeConvolution(3, input,pooling_size=pooling_size)
        self.WisePooling = GraphAttentionPooling(input, pooling_size=pooling_size)
        self.Propagate1 = Parameter(torch.FloatTensor(1024, 1024))

        self.NodeConvolution1 = NodeConvolution(3, 1024,pooling_size=pooling_size)
        self.AttentionPooling1 = GraphAttentionPooling(1024, pooling_size=pooling_size)
        self.Propagate2 = Parameter(torch.FloatTensor(1024, 512))

        self.NodeConvolution2 = NodeConvolution(3, 512, pooling_size=pooling_size)
        self.AttentionPooling2 = GraphAttentionPooling(512, pooling_size=pooling_size)
        self.Propagate3 = Parameter(torch.FloatTensor(512, 256))

        self.reset_parameters()
        self.MixtureOfExpert = MoE(256*2, nclass, nclass, hidden_size=64, noisy_gating=True, k=4)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Propagate1.size(1))
        self.Propagate1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Propagate2.size(1))
        self.Propagate2.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Propagate3.size(1))
        self.Propagate3.data.uniform_(-stdv, stdv)

    def forward(self, x, graph, device):

        adj = self.WisePooling(x)
        x = self.nodewiseconvolution(x)  # 2,256
          # 2,1024
        adj = self.get_adjacent(adj).to(device)  # 2,2

        x = adj@x@ self.Propagate1
        x = F.elu(x)

        adj = self.AttentionPooling1(x)  # 2,64
        x = self.NodeConvolution1(x)  # 2,64
        adj = self.get_adjacent(adj).to(device)  # 2,32.

        x = adj @ x @ self.Propagate2
        x = F.elu(x)

        adj = self.AttentionPooling2(x)  # 2,64
        x = self.NodeConvolution2(x)  # 2,64
        adj = self.get_adjacent(adj).to(device)  # 2,32.

        x = adj @ x @ self.Propagate3
        x = F.elu(x)

        x= x.view(-1,512)
        x = self.MixtureOfExpert(x)

        return x

    def cosine_similarity_adjacent(self, matrix1, matrix2):
        squaresum1 = torch.sum(torch.square(matrix1), dim=1)  # 1024 to 1

        squaresum2 = torch.sum(torch.square(matrix2), dim=1)  # 1024 to 1

        multiplesum = torch.sum(torch.multiply(matrix1, matrix2), dim=1)

        Matrix1DotProduct = torch.sqrt(squaresum1)
        Matrix2DotProduct = torch.sqrt(squaresum2)
        cosine_similarity = torch.div(multiplesum, torch.multiply(Matrix1DotProduct, Matrix2DotProduct))
        return cosine_similarity

    def get_adjacent(self, matrix):
        matrix_frame = matrix.shape[0]  # 4,2,1024
        AdjacentMatrix = torch.zeros( matrix_frame, matrix_frame)  # 2 X 2

        chunks = torch.chunk(matrix, matrix_frame, dim=0)
        for i in range(matrix_frame):
            for j in range(matrix_frame - i):
                AdjacentMatrix[j][i] = self.cosine_similarity_adjacent(chunks[i], chunks[j])
                if not i == j:
                    AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
        I = torch.eye(AdjacentMatrix.shape[0])

        AdjacentMatrix += I
        D_hat = torch.sum(AdjacentMatrix, dim=0)
        D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))

        return D_hat @ AdjacentMatrix @ D_hat
