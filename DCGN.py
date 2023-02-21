import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NodeConvolution(Module):

    def __init__(self, kernel=3):
        super(NodeConvolution, self).__init__()
        self.pooling_size = kernel
        self.weight1 = Parameter(torch.FloatTensor(1, kernel))
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
        for i in range(steps):
            if left_tensor is True and i == steps - 1:
                left_size = self.pooling_size - batch_size % self.pooling_size
                tensor = input[self.pooling_size * i + 0 - left_size: self.pooling_size * i + self.pooling_size - left_size]
            else:
                tensor = input[self.pooling_size * i + 0:self.pooling_size * i + self.pooling_size ]  # 3,256
            tensor_list.append(torch.squeeze(torch.mm(self.weight1, tensor)))  # 1,256
        return torch.stack(tensor_list)  # 17,256


class WisePooling(Module):

    def __init__(self):
        super(WisePooling, self).__init__()

    def forward(self, input, graph):
        steps = graph.shape[0]
        Adjacent_matrix = torch.zeros_like(input)
        for i in range(steps):
            shot_boundary = graph[i]
            target_frames = input[shot_boundary[0]:shot_boundary[1] + 1]
            summation = torch.div(torch.sum(target_frames, dim=0), shot_boundary[1] - shot_boundary[0] + 1)  # 1,1024
            Adjacent_matrix[shot_boundary[0]:shot_boundary[1] + 1] += summation

        return Adjacent_matrix


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
        for i in range(steps):
            if left_tensor is True and i == steps - 1:
                left_size = self.pooling_size - batch_size % self.pooling_size
                tensor = batch_rep[self.pooling_size * i + 0 - left_size: self.pooling_size * i + 3 - left_size]
            else:
                tensor = batch_rep[self.pooling_size * i + 0:self.pooling_size * i + 3]
            softmax = nn.functional.softmax
            att_w = softmax(self.W(tensor).squeeze(-1),dim=0).unsqueeze(-1)
            tensor_list.append(torch.sum(tensor * att_w, dim=1))

        return torch.squeeze(torch.stack(tensor_list))


def cosine_similarity_adjacent(matrix1, matrix2):
    squaresum1 = torch.sum(torch.squeeze(torch.square(matrix1)))  # 1024 to 1

    squaresum2 = torch.sum(torch.squeeze(torch.square(matrix2)))  # 1024 to 1

    multiplesum = torch.sum(torch.squeeze(torch.multiply(matrix1, matrix2)))

    Matrix1DotProduct = torch.sqrt(squaresum1)
    Matrix2DotProduct = torch.sqrt(squaresum2)
    cosine_similarity = torch.div(multiplesum, torch.multiply(Matrix1DotProduct, Matrix2DotProduct))
    return cosine_similarity  # (batch, 256x256)


def get_adjacent(matrix):
    matrix_frame = matrix.shape[0]  # 2,1024
    chunks = torch.chunk(matrix, matrix_frame, dim=0)  # 2 frames 1,1024
    AdjacentMatrix = torch.ones(matrix_frame, matrix_frame)  # (frames, frames,batch,256X256)
    for i in range(matrix_frame):
        for j in range(matrix_frame - i):
            AdjacentMatrix[i][j] = cosine_similarity_adjacent(chunks[i], chunks[j])
            if not i == j:
                AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
    I = torch.eye(AdjacentMatrix.shape[0])
    AdjacentMatrix += I
    D_hat = torch.sum(AdjacentMatrix, dim=0)
    D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))
    return D_hat @ AdjacentMatrix @ D_hat


from MOE import MoE


class DCGN(nn.Module):
    def __init__(self, input, nclass, pooling_size=3):
        super(DCGN, self).__init__()

        self.nodewiseconvolution = NodeConvolution(kernel=1)
        self.WisePooling = WisePooling()
        self.Propagate1 = Parameter(torch.FloatTensor(input, 256))

        self.NodeConvolution1 = NodeConvolution(3)
        self.AttentionPooling1 = GraphAttentionPooling(256, pooling_size=pooling_size)
        self.Propagate2 = Parameter(torch.FloatTensor(256, 64))

        self.NodeConvolution2 = NodeConvolution(3)
        self.AttentionPooling2 = GraphAttentionPooling(64, pooling_size=pooling_size)
        self.Propagate3 = Parameter(torch.FloatTensor(64, 32))

        self.reset_parameters()
        self.MixtureOfExpert = MoE(32 * 6, nclass, nclass, hidden_size=32, noisy_gating=True, k=4)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Propagate1.size(1))
        self.Propagate1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Propagate2.size(1))
        self.Propagate2.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.Propagate3.size(1))
        self.Propagate3.data.uniform_(-stdv, stdv)

    def forward(self, x, graph,device):
        # 50,1024
        x = self.nodewiseconvolution(x)  # 50,1024
        adj = self.WisePooling(x, graph)  # 50,1024
        adj = get_adjacent(adj).to(device)  # 50,50
        x = adj @ x @ self.Propagate1  # 50,256
        x = F.elu(x)

        adj = self.AttentionPooling1(x)# 17,256
        x = self.NodeConvolution1(x)  # 17,256
        adj = get_adjacent(adj).to(device)  # 17,17
        x = adj @ x @ self.Propagate2  # 17,64
        x = F.elu(x)

        adj = self.AttentionPooling2(x)  # 6,64
        x = self.NodeConvolution2(x)  # 6,64
        adj = get_adjacent(adj).to(device)  # 6,6
        x = adj @ x @ self.Propagate3  # 6,32
        x = F.elu(x)

        x = x.view(-1,192)  # 192
        x = self.MixtureOfExpert(x)

        return x



