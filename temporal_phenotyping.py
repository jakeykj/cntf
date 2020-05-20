import os
from socket import gethostname
import pickle
import time
import datetime

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

if torch.__version__ == '0.4.1':
    from torch.utils.data import Subset
else:
    from models import Subset
# from torch.utils.data.sampler import SubsetRandomSampler
from sklearn import metrics
from tensorboardX import SummaryWriter

from models import CNTF, HITF, TemporalDependency, MortalitySupervised, nonnegative_projection
from evaluations import logistic_regression, prediction_at_discharge


class TemporalPhenotyping:
    def __init__(self,
                 exp_name,
                 rank, alpha_CNTF, alpha_HITF, beta,
                 temporal_model,
                 temporal_model_params,
                 device=None,
                 tensorboard_summary_writer=None,
                 **optimizer_params):
        self.rank = rank
        self.alpha_CNTF = alpha_CNTF
        self.alpha_HITF = alpha_HITF
        self.beta = beta
        self.temporal_type = temporal_model
        self.temporal_model_params = temporal_model_params

        self.model_params = {
            'rank': rank,
            'alpha_CNTF': alpha_CNTF,
            'alpha_HITF': alpha_HITF,
            'beta': beta
        }

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        exp_date = datetime.datetime.now().strftime("%y%m%d%H")
        self.exp_id = f"{exp_date}-{exp_name}-alpha{alpha_CNTF}_{alpha_HITF}-beta{beta}" +\
                      f"-R{rank}-{gethostname()}-{os.getpid()}"
        results_dir = f'results/cntf_hitf/{self.exp_id}'
        self.results_dir = results_dir
        self.writer = tensorboard_summary_writer

        self.model_params = {
            'rank': rank,
            'weightings': (alpha_CNTF, alpha_HITF, beta),
            'exp_id': self.exp_id,
            'results_dir': results_dir
        }

        self.train_info = {}

        self.optimizer_params = {
            'temp_init_lr': 0.1, ##
            'U_init_lr': 0.1, ## 0.01
            'Ws_init_lr': 0.1,
            'decay_weight': 0.1,
            'decay_step': 50, ##
            'temporal_decay_weight': 0.5,
            'temporal_decay_step': 10
        }
        self.optimizer_params.update(optimizer_params)

    def fit(self, dataset, train_idx, batch_size,
            random_seed, n_epochs=100, pretrain_epochs=10,
            classifier=logistic_regression):
        # Prepare envs
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        # self.dataset = dataset
        self.train_idx = train_idx
        self.y_train = [dataset.labels[p] for p in train_idx]
        self.classifier = classifier

        train_data_loader = DataLoader(Subset(dataset, train_idx),
                                       batch_size=batch_size,
                                       shuffle=False,
                                       collate_fn=lambda x: x)

        if self.writer is None:
            self.writer = SummaryWriter(self.results_dir)
        _, Nl, Nm = dataset[0][0].shape
        Nd = dataset[0][2].shape[0]

        if random_seed:
            torch.manual_seed(random_seed)
        self.random_state = torch.random.get_rng_state()

        # Construct temporal model
        nlayers, nhidden, dropout = map(self.temporal_model_params.get, ['nlayers', 'nhidden', 'dropout'])
        self.temp_model = self.temporal_type(self.rank, nlayers, nhidden, dropout)
        # if self.temporal_type == 'temporal':
        #     self.temp_model = TemporalDependency()
        # elif self.temporal_type == 'supervised':
        #     self.temp_model = MortalitySupervised(self.rank, nlayers, nhidden, dropout)
        # else:
        #     raise NotImplementedError('The temporal regularization model is not supported.')


        self.train_info.update({
            'train_idx': train_idx,
            'y_train': self.y_train,
            'random_seed': random_seed,
            'random_state': self.random_state,
            'n_epochs': n_epochs,
            'pretrain_epochs': pretrain_epochs
        })

        temp_init_lr, U_init_lr, Ws_init_lr = map(self.optimizer_params.get, ['temp_init_lr', 'U_init_lr', 'Ws_init_lr'])
        decay_weight, decay_step = map(self.optimizer_params.get, ['decay_weight', 'decay_step'])
        temporal_decay_weight, temporal_decay_step = map(self.optimizer_params.get, ['temporal_decay_weight', 'temporal_decay_step'])

        # Initialization
        self.cntf = CNTF().to(self.device)
        self.hitf = HITF().to(self.device)
        self.temp_model = self.temp_model.to(self.device)

        self.Ws = [Variable(torch.rand(Tp, self.rank).to(self.device),
                            requires_grad=True) for Tp in dataset.los_days]
        self.Ul = Variable(torch.rand(Nl, self.rank).to(self.device), requires_grad=True)
        self.Um = Variable(torch.rand(Nm, self.rank).to(self.device), requires_grad=True)
        self.Ud = Variable(torch.rand(Nd, self.rank).to(self.device), requires_grad=True)

        optimizer_temp = torch.optim.Adam(self.temp_model.parameters(), lr=temp_init_lr)
        temp_lr_scheduler = StepLR(optimizer_temp, temporal_decay_step, gamma=temporal_decay_weight)
        for epoch in range(n_epochs):
            if isinstance(self.temp_model, MortalitySupervised):
                epoch_score_train = torch.FloatTensor().to(self.device)
                labels_train = []

            if epoch >= pretrain_epochs:
                temp_lr_scheduler.step()

            lr_decay = decay_weight ** (epoch // decay_step)
            epoch_total_loss, epoch_cntf_loss, epoch_hitf_loss, epoch_temporal_loss = 0, 0, 0, 0

            tic = time.time()

            for batch_idx, batch_samples in enumerate(train_data_loader):
                Xs, rx_vectors, dx_vectors, labels, indices = zip(*batch_samples)
                Ws_batch = [self.Ws[p] for p in indices]
                Ws_batch_nograd = [self.Ws[p].data for p in indices]

                # update Ul
                optimizer = torch.optim.Adam([self.Ul], lr=U_init_lr*lr_decay)
                # optimizer = torch.optim.LBFGS([self.Ul])
                def closure():
                    optimizer.zero_grad()
                    total_loss = self.alpha_CNTF * self.cntf(Xs, Ws_batch_nograd, self.Ul, self.Um.data)
                    # total_loss += 100 * torch.norm(self.Ul, 1)  # sparsity
                    total_loss.backward()
                    return total_loss
                optimizer.step(closure)
                nonnegative_projection(self.Ul)

                # update Um
                optimizer = torch.optim.Adam([self.Um], lr=U_init_lr*lr_decay)
                # optimizer = torch.optim.LBFGS([self.Um])
                def closure():
                    optimizer.zero_grad()
                    total_loss = self.alpha_CNTF * self.cntf(Xs, Ws_batch_nograd, self.Ul.data, self.Um)
                    if self.alpha_HITF > 0:
                        total_loss += self.alpha_HITF * self.hitf(rx_vectors, dx_vectors, Ws_batch_nograd, self.Ud.data, self.Um)
                    total_loss.backward()
                    return total_loss
                optimizer.step(closure)
                nonnegative_projection(self.Um)

                # update Ud
                if self.alpha_HITF > 0:
                    optimizer = torch.optim.Adam([self.Ud], lr=U_init_lr*lr_decay)
                    # optimizer = torch.optim.LBFGS([self.Ud])
                    def closure():
                        optimizer.zero_grad()
                        hitf_ll = self.hitf(rx_vectors, dx_vectors, Ws_batch_nograd, self.Ud, self.Um.data)
                        total_loss = self.alpha_HITF * hitf_ll
                        total_loss.backward()
                        return total_loss
                    optimizer.step(closure)
                    nonnegative_projection(self.Ud)

                # update Ws
                optimizer = torch.optim.Adam(Ws_batch, lr=Ws_init_lr*lr_decay)
                optimizer.zero_grad()

                total_loss = self.alpha_CNTF * self.cntf(Xs, Ws_batch, self.Ul.data, self.Um.data)
                if self.alpha_HITF > 0:
                    total_loss += self.hitf(rx_vectors, dx_vectors, Ws_batch, self.Ud.data, self.Um.data)
                if self.beta > 0 and epoch >= pretrain_epochs:
                    if isinstance(self.temp_model, TemporalDependency):
                        temporal_loss = self.temp_model(Ws_batch, self.device)
                    elif isinstance(self.temp_model, MortalitySupervised):
                        scores, temporal_loss = self.temp_model(Ws_batch, labels, self.device)
                    else:
                        raise NotImplementedError('The temporal regularization model is not supported.')
                    total_loss += temporal_loss
                total_loss.backward()
                optimizer.step()
                nonnegative_projection(*Ws_batch)

                # update temporal model
                if self.beta > 0 and epoch >= pretrain_epochs:
                    optimizer_temp.zero_grad()
                    if isinstance(self.temp_model, TemporalDependency):
                        temporal_loss = self.temp_model(Ws_batch, self.device)
                    elif isinstance(self.temp_model, MortalitySupervised):
                        scores, temporal_loss = self.temp_model(Ws_batch, labels, self.device)
                    temporal_loss.backward()
                    optimizer_temp.step()

                # Evaluate training process
                with torch.no_grad():
                    cntf_loss = self.cntf(Xs, Ws_batch, self.Ul.data, self.Um.data)
                    epoch_cntf_loss += cntf_loss.item()
                    if self.alpha_HITF > 0:
                        hitf_loss = self.hitf(rx_vectors, dx_vectors, Ws_batch_nograd, self.Ud.data, self.Um.data)
                        epoch_hitf_loss += hitf_loss.item()
                    if self.beta > 0 and epoch >= pretrain_epochs:
                        if isinstance(self.temp_model, TemporalDependency):
                            temporal_loss = self.temp_model(Ws_batch, self.device)
                        elif isinstance(self.temp_model, MortalitySupervised):
                            scores, temporal_loss = self.temp_model(Ws_batch, labels, self.device)
                            scores = torch.cat([torch.cat([s[:, :7], s[:, -1].view(1, -1)], dim=1) for s in scores], dim=0)
                            epoch_score_train = torch.cat([epoch_score_train, scores], dim=0)
                            labels_train += labels
                        epoch_temporal_loss += temporal_loss.item()
            epoch_total_loss = self.alpha_CNTF * epoch_cntf_loss + self.alpha_HITF * epoch_hitf_loss + self.beta * epoch_temporal_loss

            # Log the training process
            self.writer.add_scalar('Loss/Training-Total', epoch_total_loss, epoch+1)
            self.writer.add_scalar('Loss/Training-CNTF', epoch_cntf_loss, epoch+1)
            self.writer.add_scalar('Loss/Training-Temporal', epoch_temporal_loss, epoch+1)
            self.writer.add_scalar('Loss/Training-HITF', epoch_hitf_loss, epoch+1)

            if self.beta > 0 and epoch >= pretrain_epochs:
                for name, param in self.temp_model.named_parameters():
                    self.writer.add_histogram(name, param.data.cpu().numpy(), epoch+1)
                    self.writer.add_histogram(name + '/grad', param.grad.cpu().numpy(), epoch+1)

            self.writer.add_scalar('Sparsity/Ul', self.Ul.data[self.Ul.data > 0].shape[0]/(self.Ul.shape[0]*self.Ul.shape[1]), epoch+1)
            self.writer.add_scalar('Sparsity/Um', self.Um.data[self.Um.data > 0].shape[0]/(self.Um.shape[0]*self.Um.shape[1]), epoch+1)
            self.writer.add_scalar('Sparsity/Ud', self.Ud.data[self.Ud.data > 0].shape[0]/(self.Ud.shape[0]*self.Ud.shape[1]), epoch+1)

            if isinstance(self.temp_model, TemporalDependency):
                auc = prediction_at_discharge([self.Ws[p].data.cpu() for p in self.train_idx], self.y_train)
            elif isinstance(self.temp_model, MortalitySupervised):
                if epoch >= pretrain_epochs:
                    fpr, tpr, thresholds = metrics.roc_curve(labels_train, epoch_score_train[:, -1])
                    auc = metrics.auc(fpr, tpr)
                else:
                    auc = 0
            self.writer.add_scalar('AUC/Training-discharge', auc, epoch + 1)

            print(f'Epoch {epoch+1:3d}: loss={epoch_total_loss:.6e} | '
                  f'lrDecay={lr_decay:.2e} | '
                  f'AUC@discharge={auc:.2f} | '
                  f'time={time.time()-tic:.1f}s')
            self.save(results_name='dump', epoch=epoch+1)

        if isinstance(self.temp_model, TemporalDependency):
            self.Wtrain_discharge = torch.cat([self.Ws[p].data.cpu().sum(dim=0).reshape(1, -1) for p in self.train_idx]).numpy()
            self.Wtrain_ndays = [torch.cat([self.Ws[p].data.cpu()[:n, :].sum(dim=0).reshape(1, -1) for p in self.train_idx]).numpy() for n in range(1, 8)]
        print('Training Done.')

    def project(self, dataset, test_idx, batch_size, proj_epochs=50):
        self.test_idx = test_idx
        self.y_test = [dataset.labels[p] for p in test_idx]

        self.train_info.update({
            'test_idx': self.test_idx,
            'y_test': self.y_test
        })

        test_data_loader = DataLoader(Subset(dataset, test_idx),
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=lambda x: x)

        for epoch in range(proj_epochs):
            if isinstance(self.temp_model, MortalitySupervised):
                epoch_score_vali = torch.FloatTensor().to(self.device)
                labels_vali = []

            epoch_loss = 0
            tic = time.time()
            Ul_bar, Um_bar = self.Ul.data, self.Um.data
            for batch_idx, batch_samples in enumerate(test_data_loader):
                Xs, rx_vectors, dx_vectors, labels, indices = zip(*batch_samples)
                Ws_batch = [self.Ws[p] for p in indices]

                # Project Xs onto learned CNTF model without considering the temporal model.
                optimizer = torch.optim.Adam(Ws_batch, lr=0.1)
                optimizer.zero_grad()
                total_loss = self.alpha_CNTF * self.cntf(Xs, Ws_batch, Ul_bar, Um_bar)
                # if self.beta > 0:
                #     scores, temporal_loss = self.temp_model(Ws_batch, labels, self.device)
                #     total_loss += self.beta * temporal_loss
                total_loss.backward()
                optimizer.step()
                nonnegative_projection(*Ws_batch)

                # Evaluation of the projection loss.
                with torch.no_grad():
                    cntf_loss = self.alpha_CNTF * self.cntf(Xs, Ws_batch, Ul_bar, Um_bar)
                    if self.beta > 0 and isinstance(self.temp_model, MortalitySupervised):
                        self.temp_model.eval()
                        scores, temporal_loss = self.temp_model(Ws_batch, labels, self.device)                        
                        scores = torch.cat([torch.cat([s[:, :7], s[:, -1].view(1, -1)], dim=1) for s in scores], dim=0)
                        epoch_score_vali = torch.cat([epoch_score_vali, scores], dim=0)
                        labels_vali += labels

                epoch_loss += cntf_loss.item()
            print(f'Projection epoch {epoch+1}: loss={epoch_loss:.6e}, time={time.time()-tic:.1f}s')
            self.writer.add_scalar('Loss/Projection', epoch_loss, epoch+1)

            # Log the projection process
            if isinstance(self.temp_model, TemporalDependency):
                Wtest_discharge = torch.cat([self.Ws[p].data.cpu().sum(dim=0).reshape(1, -1) for p in test_idx]).numpy()
                Wtest_ndays = [torch.cat([self.Ws[p].data.cpu()[:n, :].sum(dim=0).reshape(1, -1) for p in test_idx]).numpy()
                               for n in range(1, 8)]
                auc_discharge = self.classifier(self.Wtrain_discharge, self.y_train, Wtest_discharge, self.y_test)
                for n in range(7):
                    auc = self.classifier(self.Wtrain_ndays[n], self.y_train, Wtest_ndays[n], self.y_test)
                    self.writer.add_scalar(f'AUC/Projection-{n+1}days', auc, epoch + 1)
            elif isinstance(self.temp_model, MortalitySupervised):
                fpr, tpr, thresholds = metrics.roc_curve(labels_vali, epoch_score_vali[:, -1], pos_label=1)
                auc_discharge = metrics.auc(fpr, tpr)
                for n in range(7):
                    fpr, tpr, thresholds = metrics.roc_curve(labels_vali, epoch_score_vali[:, n], pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    self.writer.add_scalar(f'AUC/Projection-{n+1}days', auc, epoch + 1)
            else:
                raise NotImplementedError('The temporal regularization model is not supported.')
            self.writer.add_scalar('AUC/Projection-discharge', auc_discharge, epoch + 1)
                # if self.beta > 0:
                #     Wtrain_mdays = torch.cat(
                #         [self.temp_model.predict(self.Ws[p].data, n + 1, 3).sum(dim=0).reshape(1, -1) for p in
                #          self.train_idx]).cpu().numpy()
                #     Wtest_mdays = torch.cat(
                #         [self.temp_model.predict(self.Ws[p].data, n + 1, 3).sum(dim=0).reshape(1, -1) for p in
                #          test_idx]).cpu().numpy()
                #     auc = self.classifier(self.Wtrain_ndays[n] + Wtrain_mdays, self.y_train,
                #                           Wtest_ndays[n] + Wtest_mdays, self.y_test)
                #     self.writer.add_scalar(f'AUC/Projection-{n+1}days+3predicted', auc, epoch + 1)
        print('Projection Done.')

    def save(self, results_name='finalModel', epoch=-1, temp_model_name='temp_model'):
        with open(os.path.join(self.results_dir, f'{results_name}.pkl'), 'wb') as f:
            pickle.dump({
                'Ws': [W.data.cpu() for W in self.Ws],
                'Ul': self.Ul.data.cpu(),
                'Ud': self.Ud.data.cpu(),
                'Um': self.Um.data.cpu(),
                'model_params': self.model_params,
                'train_info': self.train_info,
                'optimizer_params': self.optimizer_params,
                'epoch': epoch
            }, f, protocol=2)
        torch.save(self.temp_model, os.path.join(self.results_dir, f'{temp_model_name}.pt'))
        torch.save([W.data.cpu() for W in self.Ws], os.path.join(self.results_dir, 'Ws.pt'))
        torch.save([self.Ul.data.cpu(), self.Ud.data.cpu(), self.Um.data.cpu()], os.path.join(self.results_dir, 'phenotypes.pt'))

    # @staticmethod
    # def load_model_from_file(results_file, temp_model_file):
    #     with open(results_file, 'rb') as f:
    #         infile = pickle.load(f)
    #     model = temporalPhenotyping()
    #     model.__dict__.update({k: infile[k] for k in ['Ws', 'Ul', 'Ud', 'Um']})
    #     model.__dict__.update(infile['model_params'])
    #     model.optimizer_params = infile['optimizer_params']
    #     model.temp_model = torch.load(temp_model_file)
    #     return model
