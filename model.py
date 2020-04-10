import torch
import logging
import argparse
import numpy as np
from torch import nn, optim
from numpy import random as rd
from torch.utils.data import DataLoader
from util import SiameseNetworkDataset, init_weights, tes_vec
from neural_nets import SiameseNetwork, NETA, NETB
logging.basicConfig(format='%(message)s', level=logging.INFO)


def nsf(paras):
    cuda = torch.device('cuda:'+str(paras.gpu_id))
    len_anchor = paras.total_anchor
    anchor_all = list(range(0, len_anchor))
    len_s = paras.len_s
    len_t = paras.len_t
    node_f1 = list(range(0, len_s))
    node_f2 = list(range(0, len_t))
    feature_s = paras.feature_s
    feature_t = paras.feature_t
    dim = paras.represent_dim
    ker_size = paras.ker_size
    coefficient = paras.coefficient
    epoch = paras.epoch
    ratio = paras.train_ratio
    margin = paras.epsilon
    lr = paras.lr
    lr_step = paras.lr_step
    lr_prob = paras.lr_prob


    a_array_load = np.load(feature_s)
    a_array_tensor = torch.Tensor(a_array_load)
    b_array_load = np.load(feature_t)
    b_array_tensor = torch.Tensor(b_array_load)

    seeds = list(np.random.randint(0, 10000, 4))
    seed1 = seeds[0]
    seed2 = seeds[1]
    torch.cuda.manual_seed_all(seeds[2])
    torch.manual_seed(seeds[3])

    rd.seed(seed1)
    anchor_train = rd.choice(anchor_all, int(ratio*len_anchor))
    anchor_test = list(set(anchor_all)-set(anchor_train))
    triplet_neg = 1
    anchor_flag = 1
    anchor_train_len = len(anchor_train)
    anchor_train_a_list = anchor_train
    anchor_train_b_list = anchor_train
    input_a = []
    input_b = []
    classifier_target = torch.empty(0, 0).to(device=cuda)
    np.random.seed(seed2)
    index = 0
    while index < anchor_train_len:
        a = anchor_train_a_list[index]
        b = anchor_train_b_list[index]
        input_a.append(a)
        input_b.append(b)
        an_target = torch.ones(anchor_flag).to(device=cuda)
        classifier_target = torch.cat((classifier_target, an_target), dim=0)
        an_negs_index = list(set(node_f2) - {b})
        an_negs_index_sampled = list(np.random.choice(an_negs_index, triplet_neg, replace=False))
        an_as = triplet_neg * [a]
        input_a += an_as
        input_b += an_negs_index_sampled

        an_negs_index1 = list(set(node_f1) - {a})
        an_negs_index_sampled1 = list(np.random.choice(an_negs_index1, triplet_neg, replace=False))
        an_as1 = triplet_neg * [b]
        input_b += an_as1
        input_a += an_negs_index_sampled1

        un_an_target = torch.zeros(triplet_neg*2).to(device=cuda)
        classifier_target = torch.cat((classifier_target, un_an_target), dim=0)
        index += 1

    cosine_target = torch.unsqueeze(2*classifier_target - 1, dim=1)
    classifier_target = torch.unsqueeze(classifier_target, dim=1)

    ina = a_array_load[input_a]
    inb = b_array_load[input_b]
    ina = torch.Tensor(ina).to(device=cuda)
    inb = torch.Tensor(inb).to(device=cuda)
    tensor_dataset = SiameseNetworkDataset(ina, inb, classifier_target, cosine_target)
    data_loader = DataLoader(tensor_dataset, batch_size=56, shuffle=False)

    P, M = 0, 0
    model = SiameseNetwork(dim, ker_size, len_s, len_t).to(device=cuda)
    init_weights(model)
    neta = NETA(dim, ker_size, len_s).to(device=cuda)
    netb = NETB(dim, ker_size, len_t).to(device=cuda)
    a_array_tensor = a_array_tensor.to(device=cuda)
    b_array_tensor = b_array_tensor.to(device=cuda)
    cos = nn.CosineEmbeddingLoss(margin=0)
    optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_prob)

    for epoch in range(epoch):
        model.train()
        scheduler.step()
        train_loss = 0
        loss_reg = 0
        loss_anchor = 0
        for data_batch in data_loader:
            in_a, in_b, c, cosine = data_batch
            cosine = torch.squeeze(cosine, dim=1)
            in_a = torch.unsqueeze(in_a, dim=1).to(device=cuda)
            in_b = torch.unsqueeze(in_b, dim=1).to(device=cuda)
            h_a, h_b = model(in_a, in_b)
            loss_anchor_batch = 1*cos(h_a, h_b, cosine)
            loss_reg_batch = coefficient*(h_a.norm() + h_b.norm())
            loss = loss_reg_batch + loss_anchor_batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss_reg += loss_reg_batch.item()
            loss_anchor += loss_anchor_batch.item()

        neta_dict = neta.state_dict()
        netb_dict = netb.state_dict()
        model.cpu()
        trainmodel_dict = model.state_dict()

        trainmodel_dict_a = {k: v for k, v in trainmodel_dict.items() if k in neta_dict}
        trainmodel_dict_b = {k: v for k, v in trainmodel_dict.items() if k in netb_dict}
        neta_dict.update(trainmodel_dict_a)
        netb_dict.update(trainmodel_dict_b)
        neta.load_state_dict(neta_dict)
        netb.load_state_dict(netb_dict)

        neta.eval()
        netb.eval()
        hidden_a = neta(torch.unsqueeze(a_array_tensor, dim=1))
        hidden_b = netb(torch.unsqueeze(b_array_tensor, dim=1))

        if epoch >= epoch-30:
            PatN_t, MatN_t = tes_vec(hidden_a, hidden_b, anchor_train, anchor_test, node_f2)
            P += PatN_t
            M += MatN_t
        model.to(device=cuda)
    logging.info('%d %d %d %d %d %.4f %.1f %d %d %.3f %.3f' %
                 (seeds[0], seeds[1], seeds[2], seeds[3], ker_size, coefficient,
                  margin, ratio, dim, P/30, M/30))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--feature_s', type=str, default='data/personalized_global_features/facebook_feature.npy', help='feature of source network')
    parser.add_argument('--feature_t', type=str, default='data/personalized_global_features/twitter_feature.npy', help='feature of target network')
    parser.add_argument('--len_s', type=int, default=1792, help='the number of users in source network')
    parser.add_argument('--len_t', type=int, default=3493, help='the number of users in target network')
    parser.add_argument('--total_anchor', type=int, default=1515, help='total number of anchor links')
    parser.add_argument('--train_ratio', type=float, default=0.3, help='train ratio of anchor')
    parser.add_argument('--represent_dim', type=int, default=56, help='the dimension of representation vector')
    parser.add_argument('--epoch', type=int, default=150, help='epoch for user representation')
    parser.add_argument('--N', type=int, default=30, help='top N for Precision and MAP')
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID')
    parser.add_argument('--ker_size', type=int, default=7, help='kernel size of cnn')
    parser.add_argument('--lr', type=float, default=3, help='init represent learning rate')
    parser.add_argument('--lr_step', type=float, default=10, help='step for dynamic learning rate')
    parser.add_argument('--lr_prob', type=float, default=0.8, help='decay probability for dynamic learning rate')
    parser.add_argument('--epsilon', type=float, default=0, help='margin in Eq.12')
    parser.add_argument('--coefficient', type=float, default=0.0006, help='coefficient lambda in Eq.13')
    args = parser.parse_args()
    nsf(args)