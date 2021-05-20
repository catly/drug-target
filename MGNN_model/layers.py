import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid


class GraphConvolution(Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GAL(MessagePassing):
    def __init__(self, in_features, out_featrues):
        super(GAL, self).__init__()
        self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_featrues, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        # 定义leakyrelu激活函数
        self.leakyrelu = torch.nn.LeakyReLU()
        self.linear = torch.nn.Linear(in_features, out_featrues)

    def forward(self, x, edge_index):
        x = self.linear(x)
        N = x.size()[0]
        row, col = edge_index
        a_input = torch.cat([x[row], x[col]], dim=1)

        temp = torch.mm(a_input, self.a).squeeze()
        e = self.leakyrelu(temp)
        e_all = torch.zeros(N)
        for i in range(len(row)):
            e_all[row[i]] += math.exp(e[i])

        # f = open("atten.txt", "w")

        for i in range(len(e)):
            e[i] = math.exp(e[i]) / e_all[row[i]]
        #     f.write("{:.4f}\t {} \t{}\n".format(e[i], row[i], col[i]))
        #
        # f.close()
        return self.propagate(edge_index, x=x, norm=e)

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j
