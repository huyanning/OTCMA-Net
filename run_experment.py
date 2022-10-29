import argparse
from chocolate_main import chocolate_experiment
import numpy as np
import warnings
import torch
import torch.multiprocessing as mp
import os
import sys
from checkpoint import Logger


parser = argparse.ArgumentParser(description='Training code - Chocolate')
parser.add_argument("--data_path", type=str, default="data",
                    help="path to dataset repository")
parser.add_argument('--dataset', type=str, default="Sandiego", help='dataset name')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[150, 300, 350], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")

# candy specific parameters #
parser.add_argument('--lamb', default=10.0, type=float, help='SK lambda parameter')
parser.add_argument('--nopts', default=400, type=int, help='number of SK opts')
parser.add_argument("--feat_dim", default=512, type=int,
                    help="feature dimension")
parser.add_argument('--num_clusters', default=10, type=int,
                    help='number of prototypes')
parser.add_argument("--hidden_mlp", default=256, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument('-j', '--num_work', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument("--dump_path", type=str, default="checkpoint",
                    help="experiment dump path for checkpoints and log")
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use.')

def main():
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print(torch.cuda.is_available())
    # torch.backends.cudnn.benchmark = True
    for i in range(5):
        chocolate_experiment(i, args)

if __name__ == '__main__':
    main()
