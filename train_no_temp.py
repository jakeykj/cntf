from pathlib import Path
from functools import reduce
import time
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from models import TemporalTensorDataset, Subset
# from phenotools.evaluate_phenotypes import factor_matrices_to_excel


class CNTF(nn.Module):
    def __init__(self, Ts, Nl, Nm, rank):
        super().__init__()
        self.Ws = nn.ParameterList([nn.Parameter(torch.rand(Tp, rank), requires_grad=True) for Tp in Ts])
        self.Ul = nn.Parameter(torch.rand(Nl, rank), requires_grad=True)
        self.Um = nn.Parameter(torch.rand(Nm, rank), requires_grad=True)

    def loglikelihood(self, Xp, p):
        subs = Xp._indices()
        vals = Xp._values()
        sum_M = (self.Ws[p].sum(dim=0) * self.Ul.sum(dim=0) * self.Um.sum(dim=0)).sum()
        A = self.Ws[p][subs[0, :], :] * self.Ul[subs[1, :], :] * self.Um[subs[2, :], :]
        ll_cntf = (vals * torch.log(A.sum(dim=1).clamp(min=1e-10))).sum() - sum_M
        return ll_cntf / self.Ws[p].shape[0]

    def forward(self, Xs, idx):
        return -reduce(torch.add, [self.loglikelihood(Xs[i], p) for i, p in enumerate(idx)]) / len(Xs)

    def factors_nng(self):
        self.Ul.data[self.Ul.data < 0] = 0
        self.Um.data[self.Um.data < 0] = 0

    def coefficients_nng(self):
        for Wp in self.Ws:
            Wp.data[Wp.data < 0] = 0


if __name__ == '__main__':

    rank = 50
    batch_size = 200
    n_epochs = 200

    results_dir = Path('results/redo/notemp_R50')

    device = torch.device('cuda')
    torch.cuda.init()

    dataset = TemporalTensorDataset('data/rx_lab_dx-spt-balanced-180814', device)
    print(len(dataset))

    train_idx, test_idx, *_ = train_test_split(list(range(len(dataset.labels))),
                                               dataset.labels,
                                               train_size=0.8,
                                               random_state=75)

    _, Nl, Nm = dataset[0][0].shape

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=lambda x: x)
    cntf = CNTF(dataset.los_days, Nl, Nm, rank).to(device)
    optimizer = Adam(cntf.parameters(), lr=0.005)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    writer = SummaryWriter(results_dir)

    def load_combine_map_files(raw_file_dir, processed_file):
        lab_code2desc = pd.read_csv(os.path.join(raw_file_dir, 'labmap.csv'), index_col=0)
        rx_code2desc = pd.read_csv(os.path.join(raw_file_dir, 'rxmap.csv'), index_col=0)
        # dx_code2desc = pd.rad_csv(os.path.join(raw_file_dir, 'dxmap.csv'), index_col=0)
        dx_code2desc = pd.read_json(os.path.join(raw_file_dir, 'icd9.json')).set_index('code')
        with open(processed_file, 'rb') as f:
            infile = pickle.load(f)
            lab_idx2desc = {idx: lab_code2desc.loc[code]['label'] for code, idx in infile['labmap'].items()}
            rx_idx2code = {idx: rx_code2desc.loc[code]['drug_name'] for code, idx in infile['rxmap'].items()}
            dx_idx2code = {idx: dx_code2desc.loc[code]['title'] for code, idx in infile['dxmap'].items()}
        return lab_idx2desc, rx_idx2code, dx_idx2code

    lab_idx2desc, rx_idx2code, dx_idx2code = load_combine_map_files(r'D:\research_projects\cntf_phenotyping\data\raw-180808',
                                                                    r'D:\research_projects\cntf_phenotyping\data\processed-rx_lab_dx-balanced-180814.pkl')
    for epoch in range(n_epochs):
        scheduler.step()
        epoch_tic = time.time()
        for batch, samples in enumerate(train_loader):
            batch_start_tic = time.time()
            Xs, _, _, _, idx = zip(*samples)
            data_ready_tic = time.time()

            optimizer.zero_grad()
            loss = cntf(Xs, idx)
            loss.backward()
            optimizer.step()
            cntf.factors_nng()

        cntf.coefficients_nng()


        with torch.no_grad():
            epoch_loss = 0
            for batch, samples in enumerate(train_loader):
                Xs, _, _, _, idx = zip(*samples)
                loss = cntf(Xs, idx)
                epoch_loss += loss.item()
            writer.add_scalar('Loss/Train-CNTF', epoch_loss, epoch+1)
        epoch_time = time.time() - epoch_tic
        print(f'Epoch {epoch+1} done, loss={epoch_loss:.3e}, time={epoch_time:.1f}s')
        torch.save(cntf, results_dir/'checkpoint.pt')


    torch.save(cntf.cpu(), results_dir/'final_model.pt')
    factors = [cntf.Ul.cpu().data.numpy(), cntf.Um.cpu().data.numpy()]
    item_dicts = [('lab', lab_idx2desc), ('rx', rx_idx2code)]
    factor_matrices_to_excel(factors, item_dicts, results_dir/'phenotypes.xlsx')




