import torch
import torch.nn as nn
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, scatter_, sort_edge_index


class GraphPool(nn.Module):
    def __init__(self, hidd_dim, ratio=0.5, non_linearity=torch.sigmoid):
        super().__init__()
        self.ratio = ratio
        self.non_linearity = non_linearity
        self.vec1 = nn.Parameter(torch.zeros(
            1, hidd_dim), requires_grad=True)
        init.xavier_uniform_(self.vec1.data)

    def forward(self, ui_em, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # (batch * word_num)
        scores1 = torch.abs(torch.sum(ui_em[batch] * x, 1))
        scores2 = self.dist(x, edge_index)
        scores = scores1 + scores2
        perm = topk(scores, self.ratio, batch)
        x = x[perm] * self.non_linearity(scores[perm]).view(-1, 1)
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=scores.size(0))
        batch = batch[perm]
        return x, edge_index, edge_attr, batch, perm

    def dist(self, x, edge_index):
        edge_index_sort, _ = sort_edge_index(edge_index, num_nodes=x.size(0))
        dis_em = torch.abs(
            (x[edge_index_sort[0]] * self.vec1).sum(-1) - (x[edge_index_sort[1]] * self.vec1).sum(-1))
        dis_em = scatter_('mean', dis_em.unsqueeze(
            1), edge_index_sort[0])  # (word_num, dim)
        return dis_em.squeeze(1)


class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relation, negative_slope=0.2, dropout=0, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.relation_w = nn.Parameter(torch.Tensor(
            num_relation, in_channels))
        self.weight = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight1 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(
            torch.Tensor(in_channels, out_channels))
        self.weight3 = Parameter(
            torch.Tensor(in_channels, out_channels))

        init.xavier_uniform_(self.relation_w.data)
        init.xavier_uniform_(self.weight.data)
        init.xavier_uniform_(self.weight1.data)
        init.xavier_uniform_(self.weight2.data)
        init.xavier_uniform_(self.weight3.data)

    def forward(self, x, batch, edge_index, edge_type, size=None):
        return self.propagate(edge_index, size=size, x=x, batch=batch, edge_type=edge_type)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_type):
        w = torch.index_select(self.relation_w, 0, edge_type)
        x_iw = torch.matmul(x_i, self.weight2)
        x_jw = torch.matmul(x_j, self.weight3)
        x_r = torch.matmul(w, self.weight1)
        alpha = (x_iw * (x_jw + x_r)).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return torch.matmul(x_j, self.weight) * alpha.view(-1, 1)

    def update(self, aggr_out, x):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.channels, self.channels)
