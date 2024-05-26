import os
import parser

import torch
import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="GDELT-Few", type=str)
    args.add_argument("-path", "--data_path", default="./icews_18", type=str)
    args.add_argument("-form", "--data_form", default="Pre-Train", type=str)
    args.add_argument("-seed", "--seed", default='19981218', type=int)
    args.add_argument("-few", "--few", default=1, type=int)
    args.add_argument("-nq", "--num_query", default=3, type=int)
    args.add_argument("-metric", "--metric", default="MRR", choices=["MRR", "Hits@10", "Hits@5", "Hits@1"])

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=5e-5, type=float)
    args.add_argument("-es_p", "--early_stopping_patience", default=10, type=int)

    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)
    args.add_argument("-ckpt_epo", "--checkpoint_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1.0, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)
    args.add_argument("-abla", "--ablation", default=False, type=bool)
    args.add_argument("--dropout_input", default=0.3, type=float)
    args.add_argument("--dropout_layers", default=0.2, type=float)
    args.add_argument("--dropout_neighbors", default=0.0, type=float)
    args.add_argument("-gpu", "--device", default=0, type=int)

    args.add_argument("-prefix", "--prefix", default="exp1", type=str)
    args.add_argument("-step", "--step", default="train", type=str, choices=['train', 'test', 'dev'])
    args.add_argument("-log_dir", "--log_dir", default="log", type=str)
    args.add_argument("-state_dir", "--state_dir", default="state", type=str)
    args.add_argument("-eval_ckpt", "--eval_ckpt", default=None, type=str)
    args.add_argument("-eval_by_rel", "--eval_by_rel", default=False, type=bool)
    args.add_argument("-embed_model", "--embed_model", default="TTransE", type=str)
    args.add_argument("-max_neighbor", "--max_neighbor", default=50, type=int)

    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    if args.dataset == 'icews18':
        params['embed_dim'] = 100
    elif args.dataset == 'GDELT-Few':
        params['embed_dim'] = 50
    args = parser.parse_args()
    if not os.path.exists('models'):
        os.mkdir('models')
    args.save_path = 'models/' + args.prefix
    params['device'] = torch.device('cuda:'+str(args.device))

    return params


