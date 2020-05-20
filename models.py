import os
import pickle
from functools import reduce

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TemporalTensorDataset(Dataset):
    """
    Temporal Tensor Dataset. Each patient is represented by a temporal tensor,
    with dimensions of time, labtests and medications.
    """
    def __init__(self, data_dir, device):
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir, 'list.pkl'), 'rb') as f:
            infile = pickle.load(f)
        self.hadm_id_list, self.los_days, labels = zip(*infile)
        self.labels = [1 if label == 1 else 0 for label in labels]
        # load all
        with open(os.path.join(data_dir, 'data.pkl'), 'rb') as f:
            data = pickle.load(f)
        self.spt = [torch.sparse.FloatTensor(hadm['subs'], hadm['vals'], hadm['size']).to(device) for hadm in data]
        self.rx_vectors = [hadm['rx_vector'].to(device) for hadm in data]
        self.dx_vectors = [hadm['dx_vector'].to(device) for hadm in data]

    def __len__(self):
        return len(self.hadm_id_list)

    def __getitem__(self, idx):
        return self.spt[idx], self.rx_vectors[idx], self.dx_vectors[idx], self.labels[idx], idx

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class CNTF(nn.Module):
    def loglikelihood(self, Xp, Wp, Ul, Um):
        subs = Xp._indices()
        vals = Xp._values()
        sum_M = (Wp.sum(dim=0) * Ul.sum(dim=0) * Um.sum(dim=0)).sum()
        A = Wp[subs[0, :], :] * Ul[subs[1, :], :] * Um[subs[2, :], :]
        ll_cntf = (vals * torch.log(A.sum(dim=1).clamp(min=1e-10))).sum() - sum_M
        return ll_cntf / Wp.shape[0]

    def forward(self, Xs, Ws, Ul, Um):
        return -reduce(torch.add, [self.loglikelihood(Xp, Wp, Ul, Um) for Xp, Wp in zip(Xs, Ws)])


class HITF(nn.Module):
    def hitf_loglikelihood(self, rx_vector, dx_vector, Wp, Ud, Um):
        Mhat = Wp.sum(dim=0) @ torch.diag(Ud.sum(dim=0)) @ Um.t()
        Dhat = Wp.sum(dim=0) @ torch.diag(Um.sum(dim=0)) @ Ud.t()
        ll_M = -Mhat + rx_vector * torch.log(Mhat.clamp(min=1e-10))
        ll_Dprime = -Dhat + dx_vector * (Dhat + torch.log((1-torch.exp(-Dhat)).clamp(min=1e-10)))
        ll_hitf = ll_M.sum() + ll_Dprime.sum()
        return ll_hitf / Wp.shape[0]

    def forward(self, rx_vectors, dx_vectors, Ws, Ud, Um):
        return -reduce(torch.add, [self.hitf_loglikelihood(rx, dx, Wp, Ud, Um) for Wp, rx, dx in zip(Ws, rx_vectors, dx_vectors)])


def nonnegative_projection(*var):
    for X in var:
        X.data[X.data < 0] = 0


class TemporalDependency(nn.Module):
    def __init__(self, rank, nlayers, nhidden, dropout):
        super(TemporalDependency, self).__init__()

        self.nlayers = nlayers
        self.nhid = nhidden

        self.rnn = nn.LSTM(input_size=rank,
                           hidden_size=nhidden,
                           num_layers=nlayers,
                           dropout=dropout,
                           batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(nhidden, rank),
            nn.ReLU()
        )

        # self.decoder = nn.Linear(nhidden, rank)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, Ws, device):
        train_loss = 0.0
        for Wp in Ws:
            inputs, targets = Wp[:-1, :], Wp[1:, :]  # seq_len x n_dim
            seq_len, n_dims = inputs.size()

            hidden = self.init_hidden(1)
            # seq_len x n_dims --> 1 x seq_len x n_dims
            outputs, _ = self.rnn(inputs.unsqueeze(0), hidden)
            logits = self.decoder(outputs.contiguous().view(-1, self.nhid))
            loss = self.loss(logits, targets)
            train_loss += loss
        return train_loss

    def init_hidden(self, batch_sz):
        size = (self.nlayers, batch_sz, self.nhid)
        weight = next(self.parameters())
        return (weight.new_zeros(*size),
                weight.new_zeros(*size))

    # def predict(self, input, n_past, m_future):
    #     # n_factor --> batch_size
    #     batch_sz = input.size(1)
    #
    #     hidden = self.init_hidden(batch_sz)
    #     # n_factor * time
    #     input = input.t()
    #     input = input[:, :n_past]
    #     input = self.encoder(input.unsqueeze(2))
    #     outputs, hidden = self.rnn(input, hidden)
    #     logits = self.decoder(outputs.contiguous().view(-1, outputs.size(2)))
    #
    #     outputs = []
    #     if n_past == 1:
    #         future_inputs = logits
    #     else:
    #         future_inputs = logits[:batch_sz, :]
    #
    #     outputs.append(future_inputs.t())
    #     for m in range(m_future - 1):
    #         inputs = future_inputs.unsqueeze(2)
    #         inputs = self.encoder(inputs)
    #         pred, hidden = self.rnn(inputs, hidden)
    #         pred = self.decoder(pred.view(-1, pred.size(2)))
    #         if m != m_future - 2:
    #             future_inputs = pred
    #         outputs.append(pred.t())
    #     return torch.stack(outputs, 0).data

    def loss(self, input, target):
        return torch.mean((input - target) ** 2)


class MortalitySupervised(nn.Module):
    def __init__(self, ninp, nlayers, nhid, dropout):
        # ninp: num_factors
        super(MortalitySupervised, self).__init__()

        self.nlayers = nlayers
        self.nhid = nhid

        self.rnn = nn.LSTM(input_size=ninp,
                           hidden_size=nhid,
                           num_layers=nlayers,
                           dropout=dropout,
                           batch_first=True)
        self.decoder = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(nhid, 1),
            nn.Sigmoid())
        
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def forward(self, Ws, labels, device):
        train_loss = 0.0
        scores_batch = []
        
        for Wp, label in zip(Ws, labels):
            seq_len, n_features = Wp.size()
            hidden = self.init_hidden(1)
            outputs, _ = self.rnn(Wp.unsqueeze(0), hidden)

            score = self.decoder(outputs.contiguous().view(-1, self.nhid)).view(1, -1)
            label = label * torch.ones(1, seq_len, device=device)
            day_weights = torch.FloatTensor([0] + [1/n for n in range(1, seq_len)]).to(device)
            criterion = nn.BCELoss(weight=day_weights)
            # score = self.decoder(outputs.contiguous().view(-1, self.nhid))[-1]
            
            print(score)
            print(label)
            loss_batch = criterion(score, label)

            # loss = self.loss(score, label)
            train_loss += loss_batch

            scores_batch.append(score)
        return scores_batch, train_loss

    def init_hidden(self, batch_sz):
        size = (self.nlayers, batch_sz, self.nhid)
        weight = next(self.parameters())
        return (weight.new_zeros(*size),
                weight.new_zeros(*size))

    def loss(self, input, target, weight=None):
        # loss = - (target * torch.log(input)) + ((1 - target) * torch.log(1 - input))
        neg_abs = - torch.abs(input)
        loss = torch.clamp(input, min=0) - input * target + torch.log(1 + torch.exp(neg_abs))
        if weight is not None:
            loss = weight.view(1, -1) * loss
        return torch.mean(loss)


# class AutoRegressive(nn.Module):
#     def __init__(self, rank, order=3):
#         super().__init__()
#         self.rank = rank
#         self.order = order
#         self.temp_coef = nn.Parameter(torch.rand(order, rank), requires_grad=True)
#         self.bias = nn.Parameter(torch.rand(rank), requires_grad=True)
#
#     def forward(self, Ws, device):
#         outputs = []
#         for W in Ws:
#             output = torch.zeros(W.shape[0]-self.order, self.rank).cuda()
#             for n in range(0, W.shape[0]-self.order):
#                 output[n, :] = (self.temp_coef * W[n:n+self.order, :]).sum(dim=0) + self.bias
#             outputs.append(output)
#         loss = self.loss(outputs, Ws)
#         return loss
#
#     def predict(self, input, n_past, m_future):
#         input = input[:n_past, :]
#         output = torch.zeros(m_future, input.shape[1])
#         tmp = input[-self.order:, :]
#         for m in range(m_future):
#             out = (self.temp_coef * tmp).sum(dim=0) + self.bias
#             output[m, :] = out
#             tmp = torch.cat([tmp[1:, :], out.reshape(1, -1)], dim=0)
#         return output
#
#     def loss(self, input, target):
#         return reduce(torch.add, [((x - y[self.order:, :]) ** 2).mean() for x, y in zip(input, target)])


