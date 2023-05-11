

import argparse
import torch

def Configs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', dest='datapath', type=str, 
                        default='your dataset path',
                        help='path to dataset')
    parser.add_argument('--dataset', dest='dataset', type=str, default='clive',
                        help='Support datasets: clive|koniq|fblive|live|csiq|tid2013')
    parser.add_argument('--svpath', dest='svpath', type=str,
                        default='your save path',
                        help='the path to save the info')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=50,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=50,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
                        help='Epochs for training')
    parser.add_argument('--seed', dest='seed', type=int, default=1,
                        help='for reproducing the results')
    parser.add_argument('--vesion', dest='vesion', type=int, default=1,
                        help='vesion number for saving')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, 
                        help='Crop size for training & testing image patches')
    parser.add_argument('--droplr', dest='droplr', type=int, default=5,
                        help='drop lr by every x iteration')   
    parser.add_argument('--gpunum', dest='gpunum', type=str, default='1',
                        help='the id for the gpu that will be used')

    return parser.parse_args()
    
    
if __name__ == '__main__':
    config = Configs()
    for arg in vars(config):
        print(arg, getattr(config, arg))
        


    