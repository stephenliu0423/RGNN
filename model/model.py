import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.data import Data
import torch_geometric.data as gda
from torch_geometric.nn import global_max_pool as gmp
from layer import GraphPool
from layer import GAT


class RGNN(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.user_embedding = nn.Embedding(config['n_users'], args.dim)
        self.item_embedding = nn.Embedding(config['n_items'], args.dim)
        self.word_embedding = nn.Embedding(config['n_words'], args.word_dim)
        self.agg_u = nn.Linear(
            args.hidd_dim, args.dim)
        self.agg_i = nn.Linear(
            args.hidd_dim, args.dim)
        self.conv_u = nn.ModuleList(
            [GAT(args.word_dim, args.hidd_dim, num_relation=4)])
        self.conv_i = nn.ModuleList(
            [GAT(args.word_dim, args.hidd_dim, num_relation=4)])
        for _ in range(args.num_layers - 1):
            self.conv_u.append(
                GAT(args.hidd_dim, args.hidd_dim, num_relation=4))
            self.conv_i.append(
                GAT(args.hidd_dim, args.hidd_dim, num_relation=4))
        self.trans_u = nn.ModuleList(
            [nn.Linear(args.dim, args.hidd_dim) for _ in range(args.num_layers)])
        self.trans_i = nn.ModuleList(
            [nn.Linear(args.dim, args.hidd_dim) for _ in range(args.num_layers)])
        self.trans_w = nn.ModuleList(
            [nn.Linear(args.hidd_dim, args.dim) for _ in range(args.num_layers)])
        self.interaction_u = nn.Linear(args.dim, args.dim)
        self.interaction_i = nn.Linear(args.dim, args.dim)
        self.fm2 = FM_Layer(args, config)
        self.pool = GraphPool(args.hidd_dim)
        self.Batch = gda.Batch()
        self.Dropout = nn.Dropout(args.dropout)
        init.xavier_uniform_(self.user_embedding.weight)
        init.xavier_uniform_(self.item_embedding.weight)
        init.xavier_uniform_(self.word_embedding.weight)

    def forward(self, uid_batch, iid_batch, u_nodes, u_adj_ind, u_adj_tp, i_nodes, i_adj_ind, i_adj_tp):
        self.u_e = self.user_embedding(uid_batch)
        self.i_e = self.item_embedding(iid_batch)
        u_temp, i_temp = [], []
        for i in range(len(u_nodes)):
            u_x = self.word_embedding(u_nodes[i])
            u_temp.append(
                Data(x=u_x, edge_index=u_adj_ind[i], edge_attr=u_adj_tp[i].unsqueeze(1)))
            i_x = self.word_embedding(i_nodes[i])
            i_temp.append(
                Data(x=i_x, edge_index=i_adj_ind[i], edge_attr=i_adj_tp[i].unsqueeze(1)))
        u_graph = self.Batch.from_data_list(u_temp)
        i_graph = self.Batch.from_data_list(i_temp)
        u_em = [t.relu(self.trans_u[i](self.u_e))
                for i in range(self.args.num_layers)]
        i_em = [t.relu(self.trans_i[i](self.i_e))
                for i in range(self.args.num_layers)]
        u_pool_e = self.conv_pool(u_em, u_graph, self.conv_u)
        i_pool_e = self.conv_pool(i_em, i_graph, self.conv_i)
        user_rep = t.relu(self.interaction_u(self.u_e))
        item_rep = t.relu(self.interaction_i(self.i_e))
        user_vc = t.cat((user_rep, u_pool_e), -1)
        item_vc = t.cat((item_rep, i_pool_e), -1)
        pre_rate = self.fm2(user_vc, item_vc, uid_batch, iid_batch)
        return pre_rate

    def conv_pool(self, ui_em, graph, conv_ui):
        pool_e = []
        review_e, edge_index, edge_attr, batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch.cuda()
        for i in range(self.args.num_layers):
            review_e = conv_ui[i](
                review_e, batch, edge_index, edge_attr.squeeze())
            review_e, edge_index, edge_attr, batch, _ = self.pool(
                ui_em[i], review_e, edge_index, edge_attr, batch)
            out = gmp(review_e, batch)
            out = t.relu(self.trans_w[i](out))
            pool_e.append(out)
        out_e = t.cat(pool_e, -1)  # (batch, layer_num*dim)
        return out_e


class FM_Layer(nn.Module):
    def __init__(self, args, config):
        super(FM_Layer, self).__init__()
        input_dim = (args.num_layers + 1) * args.dim * 2
        # input_dim = args.dim * 2
        self.linear = nn.Linear(input_dim, 1, bias=False)
        self.V = nn.Parameter(
            t.zeros(input_dim, input_dim), requires_grad=True)
        self.bias_u = nn.Parameter(
            t.zeros(config['n_users'], requires_grad=True))
        self.bias_i = nn.Parameter(
            t.zeros(config['n_items'], requires_grad=True))
        self.bias = nn.Parameter(t.zeros(1, requires_grad=True))

        init.xavier_uniform_(self.V.data)

    def fm_layer(self, user_em, item_em, uid, iid):
        # linear_part: batch * 1 * input_dim
        x = t.cat((user_em, item_em), -1).unsqueeze(1)
        linear_part = self.linear(x).squeeze()
        batch_size = len(x)
        V = t.stack((self.V,) * batch_size)
        # batch * 1 * input_dim
        interaction_part_1 = t.bmm(x, V)  # (batch, 1, input_ dim)
        interaction_part_1 = t.pow(interaction_part_1, 2)
        interaction_part_2 = t.bmm(t.pow(x, 2), t.pow(V, 2))
        mlp_output = 0.5 * \
            t.sum((interaction_part_1 - interaction_part_2).squeeze(1), -1)
        rate = linear_part + mlp_output + \
            self.bias_u[uid] + self.bias_i[iid] + self.bias
        return rate

    def forward(self, user_em, item_em, uid, iid):
        return self.fm_layer(user_em, item_em, uid, iid).view(-1)
