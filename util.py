from torch.nn import init
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class SiameseNetworkDataset(Dataset):

    def __init__(self, a, b, c, co):
        self.a = a
        self.b = b
        self.c = c
        self.co = co

    def __getitem__(self, index):
        return self.a[index], self.b[index], self.c[index], self.co[index]

    def __len__(self):
        return self.a.size()[0]


def init_weights(net):
    con_w1 = ['cnn1.0.weight', 'cnn3.2.weight']
    con_w2 = ['cnn3.0.weight', 'cnn1.2.weight']
    norm_w = ['cnn1.1.weight', 'cnn1.3.weight', 'cnn3.1.weight', 'cnn3.3.weight']
    fc14_w = ['fc1.0.weight', 'fc4.0.weight']
    fc23_w = ['fc2.0.weight', 'fc3.0.weight']
    fc_bias = ['fc1.0.bias', 'fc2.0.bias', 'fc3.0.bias', 'fc4.0.bias']
    for name, params in net.named_parameters():
        if name in con_w1:
            init.xavier_normal_(params)
        elif name in con_w2:
            init.xavier_normal_(params)
        elif name in norm_w:
            params.data.normal_()
        elif name in fc14_w:
            init.xavier_normal_(params)
        elif name in fc23_w:
            init.xavier_normal_(params)
        elif name.find('bias') != -1:
            if name in fc_bias:
                params.data.fill_(1)
            else:
                params.data.fill_(0)


def calculate_metric(num_large, N):
    if num_large >= N:
        return 0, 0, 0
    elif num_large == 0:
        return 1, 1, 1
    else:
        return 1, (N - num_large) / N, 0


def tes_vec(h_a, h_b, anchor_train, anchor_test, n_b):
    lens = len(anchor_test)
    anchor_a_list = anchor_test
    anchor_b_list = anchor_test
    known_b_list = anchor_train
    test_user_b = list(set(n_b)-set(known_b_list))
    vec_test_b = h_b[test_user_b]
    index, PatN, MatN = 0, 0.0, 0.0
    while index < lens:
        an_a = torch.unsqueeze(h_a[anchor_a_list[index]], dim=0)
        an_b = torch.unsqueeze(h_b[anchor_b_list[index]], dim=0)
        an_sim = F.cosine_similarity(an_a, an_b).item()
        un_an_sim = F.cosine_similarity(an_a, vec_test_b)
        larger_than_anchor = un_an_sim >= an_sim
        num_large_than_anchor = int(larger_than_anchor.sum().item())
        patN, matN, _ = calculate_metric(num_large_than_anchor, 30)
        PatN += patN
        MatN += matN
        index += 1
    PatN_t, MatN_t = PatN/lens, MatN/lens
    return PatN_t, MatN_t

