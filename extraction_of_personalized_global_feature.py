import csv
import argparse
import networkx as nx
import numpy as np
from tqdm import tqdm


def read_edge_pair(path):
    with open(path) as df:
        d = csv.reader(df, delimiter=' ')
        data = [row for row in d]
    edge_pair = set()
    while len(data)>0:
        row = tuple(map(eval, data.pop(0)))
        edge_pair.add(row)
    return edge_pair


def cons_net(paras):
    path = paras.data_path
    net = read_edge_pair(path)
    G1 = nx.Graph()
    G1.add_edges_from(net)
    return G1


def save_feature(feature, paras):
    path = paras.save_path
    np.save(path + '_features', feature)


def feature_ex(paras):
    G = cons_net(paras)
    prob = paras.restart_probability
    nodes = sorted(list(G.nodes()))
    l = len(nodes)
    PageRank_array = None
    for n in tqdm(nodes):
        nl = [0]*l
        nl[n] = prob
        personal = {k : v for k, v in enumerate(nl)}
        dic = nx.pagerank(G, personalization=personal)
        kv = sorted(dic.items(), key=lambda item: item[0])
        pr_lis = [[v[1] for v in kv]]
        if PageRank_array is None:
            PageRank_array = np.array(pr_lis)
        else:
            PageRank_array = np.concatenate((PageRank_array, np.array(pr_lis)), axis=0)
    save_feature(PageRank_array, paras)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data_path', required=True, type=str, default='',
                        help='network structural with adj_list format')
    parser.add_argument('--save_path', required=True, type=str, default='',
                        help='extracted feature path')
    parser.add_argument('--restart_probability', required=True, type=float, default=0.6)

    args = parser.parse_args()
    feature_ex(args)

