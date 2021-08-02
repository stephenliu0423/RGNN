import numpy as np
import pickle
import sys
import itertools as itl


class GraphData():
    def __init__(self, args):
        # self.path = args.path
        self.path = '../data/' + args.dataset + '/'
        self.args = args
        train_file = self.path + 'data.train'
        eval_file = self.path + 'data.eval'
        test_file = self.path + 'data.test'
        para_file = self.path + 'data.para'
        user_graphs_file = self.path + 'data.user_graphs'
        item_graphs_file = self.path + 'data.item_graphs'
        self.train_data = self.load_data(train_file)
        self.eval_data = self.load_data(eval_file)
        self.test_data = self.load_data(test_file)
        self.para_data = self.load_data(para_file)
        self.u_graphs = self.load_data(user_graphs_file)
        self.i_graphs = self.load_data(item_graphs_file)
        self.statistic_ratings()
        print('number of users:', self.n_users)
        print('number of items:', self.n_items)
        print('number of words:', self.word_num)

    def load_data(self, file):
        file_path = open(file, 'rb')
        data = pickle.load(file_path)
        file_path.close()
        return data

    def statistic_ratings(self):
        self.n_users = self.para_data['user_num']
        self.n_items = self.para_data['item_num']
        self.word2id = self.para_data['vocab']
        self.word_num = len(self.word2id)
        self.train_length = self.para_data['train_length']
        self.eval_length = self.para_data['eval_length']
        self.test_length = self.para_data['test_length']

    def generate_batch(self, data_type='train'):
        if data_type == 'train':
            np.random.shuffle(self.train_data)
            data = self.train_data
        elif data_type == 'eval':
            data = self.eval_data
        elif data_type == 'test':
            data = self.test_data
        else:
            sys.exit()
        iter_num = int(len(data) / self.args.batch_size) + 1
        for batch_num in range(iter_num):
            start_index = batch_num * self.args.batch_size
            end_index = min((batch_num + 1) * self.args.batch_size, len(data))
            batch = data[start_index:end_index]
            uid_batch, iid_batch, rate_batch = zip(*batch)
            u_nodes, i_nodes = [], []
            u_adj_ind, i_adj_ind = [], []
            u_edge_type, i_edge_type = [], []
            for i, user in enumerate(uid_batch):
                adj_ind, adj_val, node_index = self.u_graphs[user]
                u_adj_ind.append(adj_ind)
                u_nodes.append(node_index)
                u_edge_type.append(adj_val)
            for i, item in enumerate(iid_batch):
                adj_ind, adj_val, node_index = self.i_graphs[item]
                i_adj_ind.append(adj_ind)
                i_nodes.append(node_index)
                i_edge_type.append(adj_val)
            user_info = (itl.chain(u_nodes), itl.chain(
                u_adj_ind), itl.chain(u_edge_type))
            item_info = (itl.chain(i_nodes), itl.chain(
                i_adj_ind), itl.chain(i_edge_type))
            yield uid_batch, iid_batch, rate_batch, user_info, item_info
