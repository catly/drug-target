import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GAL
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2= GraphConvolution(nhid, nhid2)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(SFGCN, self).__init__()
        self.SGAT1 = GAT(nfeat, nhid1, nhid2, dropout)
        self.SGAT2 = GAT(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, 16),
            nn.Tanh(),
            nn.Linear(16, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj, asadj, afadj):
        emb1 = self.SGAT1(x, asadj)  # Special_GAT out1 -- sadj structure graph
        com1 = self.CGCN(x, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x, fadj)  # Common_GCN out2 -- fadj feature graph
        emb2 = self.SGAT2(x, afadj)  # Special_GAT out2 -- fadj feature graph
        Xcom = (com1 + com2) / 2

        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        emb, att = self.attention(emb)
        output = self.MLP(emb)
        return output

class GATLay(torch.nn.Module):
    def __init__(self, in_features, hid_features, out_features, n_heads):
        super(GATLay, self).__init__()
        self.attentions = [GAL(in_features, hid_features) for _ in
                           range(n_heads)]
        self.out_att = GAL(hid_features * n_heads, out_features)

    def forward(self, x, edge_index, dropout):
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return F.softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, nfeat, nhid, nhid2, dropout):
        super(GAT, self).__init__()
        self.gatlay = GATLay(nfeat, nhid, nhid2, 4)
        self.dropout = dropout

    def forward(self, x, adj):
        edge_index = adj
        x = self.gatlay(x, edge_index, self.dropout)
        return x
