import argparse
import warnings

import torch
from sklearn.model_selection import train_test_split

from models import TemporalDependency, MortalitySupervised, TemporalTensorDataset
from temporal_phenotyping import TemporalPhenotyping


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Path of the data to be used.')

    parser.add_argument('--name', '-n', default='', help='Name of the experiment')
    parser.add_argument('--rank', type=int, default=50, help='The CP rank of the tensor factorization.')
    parser.add_argument('--alpha_CNTF', type=float, default=1, help='The weighting parameter for the CNTF loss.')
    parser.add_argument('--alpha_HITF', type=float, default=1, help='The weighting parameter for the HITF loss.')
    parser.add_argument('--beta', type=float, default=100, help='The weighting parameter for the temporal regularization.')

    parser.add_argument('--batchsize', type=int, default=50, help='Batch size')
    parser.add_argument('--nepochs', type=int, default=50, help='Number of epochs for training.')
    parser.add_argument('--projepochs', type=int, default=30, help='Number of epochs for projecting the test data.')
    parser.add_argument('--init_Wslr', type=float, default=0.3, help='Initial learning rate for Ws')
    parser.add_argument('--init_Ulr', type=float, default=0.05, help='Initial learning rate for Ud, Ul & Um')
    parser.add_argument('--decay_step', type=int, default=15, help='Step for learning rate decay')
    parser.add_argument('--decay_weight', type=float, default=0.5, help='Weight for learning rate decay')

    parser.add_argument('--tempmodel', type=str, default='temporal', help='Temporal Model to be used.')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of layers in RNN')
    parser.add_argument('--nhidden', type=int, default=200, help='Number of hidden units in RNN')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout of LSTM model')
    # parser.add_argument('--arorder', type=int, default=3, help='Order of autoregressive model')
    parser.add_argument('--pretrain', type=int, default=10, help='Number of epochs before joining the temporal model')

    parser.add_argument('--random', type=int, help='Random seed to be used.')
    parser.add_argument('--nocuda', action='store_true', help='Do not use GPU even it is available.')
    parser.add_argument('--logfile', type=str, help='Directory where the tensorboard log will be stored.')
    args = parser.parse_args()

    random_state = args.random

    if args.nocuda and torch.cuda.is_available():
        warnings.warn('GPU is available but not used. Remove \'--nocuda\' argument to use GPU.')
    if not args.nocuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.init()
    else:
        device = torch.device('cpu')

    if args.tempmodel not in ['temporal', 'supervised']:
        raise NotImplementedError('The temporal regularization model is not supported.')
    temporal_model = TemporalDependency if args.tempmodel == 'temporal' else MortalitySupervised
    temporal_model_params = {'nlayers': args.nlayers, 'nhidden': args.nhidden, 'dropout': args.dropout}

    dataset = TemporalTensorDataset(args.datapath, device)
    train_idx, test_idx, *_ = train_test_split(list(range(len(dataset.labels))),
                                               dataset.labels,
                                               train_size=0.8,
                                               test_size=0.2,
                                               random_state=args.random)

    model = TemporalPhenotyping(exp_name=args.name,
                                rank=args.rank,
                                alpha_CNTF=args.alpha_CNTF,
                                alpha_HITF=args.alpha_HITF,
                                beta=args.beta,
                                temporal_model=temporal_model,
                                temporal_model_params=temporal_model_params,
                                device=device)

    model.fit(dataset=dataset, train_idx=train_idx, batch_size=args.batchsize,
              random_seed=args.random, pretrain_epochs=args.pretrain, n_epochs=args.nepochs)
    model.project(dataset=dataset, test_idx=test_idx, batch_size=args.batchsize, proj_epochs=args.projepochs)

    model.save()


