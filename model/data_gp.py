import os
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords
import pickle
import scipy.sparse as sp
import itertools
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import networkx as nx
import sys

data_type = sys.argv[1]
ps = PorterStemmer()
# tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stopWords = set(stopwords.words('english'))
tags = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'PRP']


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


class data_process(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def data_review(self, ui_reviews):
        def clean_str(string):
            string = re.sub(r"[^A-Za-z]", " ", string)
            tokens = [w for w in word_tokenize(string.lower())]
            return tokens
        ui_text = {}
        for ui, reviews in ui_reviews.items():
            for review in reviews:
                s = clean_str(review)
                if int(ui) in ui_text:
                    ui_text[int(ui)].append(s)
                else:
                    ui_text[int(ui)] = [s]
        return ui_text

    def data_load(self, data):
        uid, iid, rate = [], [], []
        for line in data:
            line = line.split(',')
            uid.append(int(line[0]))
            iid.append(int(line[1]))
            rate.append(float(line[2]))
        return uid, iid, rate

    def get_edges(self, sentence, c):
        Windsize = 3
        edges_list = set()
        for i, word1 in enumerate(sentence):
            if word1 in stopWords or len(word1) <= 2:
                continue
            else:
                if word1 not in self.word2id:
                    self.word2id[word1] = self.w_cnt
                    self.w_cnt += 1
                word1_index = self.word2id[word1]
                c.update([word1_index])
                edges_list.add((word1_index, word1_index, 1))
                if i == len(sentence) - 1:
                    break
                else:
                    for j, word2 in enumerate(sentence[i + 1:i + Windsize]):
                        if word2 not in stopWords and len(word2) > 2:
                            if word2 not in self.word2id:
                                self.word2id[word2] = self.w_cnt
                                self.w_cnt += 1
                            word2_index = self.word2id[word2]
                            edges_list.add((word1_index, word2_index, 3))
                            edges_list.add((word2_index, word1_index, 2))
        return list(edges_list), c

    def generate_graph(self):
        def g_graph(text):
            num_node = 300
            reviews_graphs = {}
            for i in text.keys():
                G = nx.DiGraph()
                reviews = text[i]
                c = Counter()
                for review in reviews:
                    edges_list, c = self.get_edges(review, c)
                    # edges_list = self.get_edges(review, 1)
                    G.add_weighted_edges_from(edges_list)
                node_list = np.array([x[0]
                                      for x in c.most_common(num_node)])
                adj_matrix = nx.to_scipy_sparse_matrix(G, nodelist=node_list)
                index_val = sp.find(adj_matrix)
                edg_index = np.array([index_val[1], index_val[0]])
                edg_value = np.array(index_val[2])
                reviews_graphs[i] = (edg_index, edg_value, node_list)
            return reviews_graphs
        self.word2id = dict()
        self.w_cnt = 0
        u_graphs = g_graph(self.u_text)
        i_graphs = g_graph(self.i_text)
        assert len(self.word2id) == self.w_cnt
        return u_graphs, i_graphs

    def process_d(self):
        prodata = 'pro_data'
        train_data = open(os.path.join(
            self.data_dir + prodata + '/data_train.csv'), 'r')
        test_data = open(os.path.join(
            self.data_dir + prodata + '/data_test.csv'), 'r')
        valid_data = open(os.path.join(
            self.data_dir + prodata + '/data_valid.csv'), 'r')
        user_reviews = pickle.load(
            open(os.path.join(self.data_dir + prodata + '/user_review'), 'rb'))
        item_reviews = pickle.load(
            open(os.path.join(self.data_dir + prodata + '/item_review'), 'rb'))
        # shuffle data and select train set,test set and validation set
        print('load rating data...')
        uid_train, iid_train, rate_train = self.data_load(train_data)
        uid_valid, iid_valid, rate_valid = self.data_load(valid_data)
        uid_test, iid_test, rate_test = self.data_load(test_data)
        num_rating = len(rate_train) + len(rate_test) + len(rate_valid)
        print('splitting reviews...')
        self.u_text = self.data_review(user_reviews)
        self.i_text = self.data_review(item_reviews)
        self.user_num = len(self.u_text)
        self.item_num = len(self.i_text)
        print('generating graph of reviews')
        u_graphs, i_graphs = self.generate_graph()
        print('number of users:', self.user_num)
        print('number of items:', self.item_num)
        print('number of ratings:', num_rating)
        print('number of words', len(self.word2id))
        para = {}
        para['user_num'] = self.user_num
        para['item_num'] = self.item_num
        para['rate_num'] = num_rating
        para['vocab'] = self.word2id
        para['train_length'] = len(rate_train)
        para['eval_length'] = len(rate_valid)
        para['test_length'] = len(rate_test)
        print('write begin')
        d_train = list(zip(uid_train, iid_train, rate_train))
        d_valid = list(zip(uid_valid, iid_valid, rate_valid))
        d_test = list(zip(uid_test, iid_test, rate_test))
        train_path = open(os.path.join(
            self.data_dir, 'data.train'), 'wb')
        pickle.dump(d_train, train_path)
        valid_path = open(os.path.join(
            self.data_dir, 'data.eval'), 'wb')
        pickle.dump(d_valid, valid_path)
        test_path = open(os.path.join(
            self.data_dir, 'data.test'), 'wb')
        pickle.dump(d_test, test_path)
        para_path = open(os.path.join(
            self.data_dir, 'data.para'), 'wb')
        pickle.dump(para, para_path)
        u_graph_path = open(os.path.join(
            self.data_dir, 'data.user_graphs'), 'wb')
        pickle.dump(u_graphs, u_graph_path)
        i_graph_path = open(os.path.join(
            self.data_dir, 'data.item_graphs'), 'wb')
        pickle.dump(i_graphs, i_graph_path)
        print('done!')


if __name__ == '__main__':
    # np.random.seed(2019)
    # random.seed(2019)
    Data_process = data_process('../data/' + data_type + '/')
    Data_process.process_d()
