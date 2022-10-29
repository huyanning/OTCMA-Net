# coding=utf-8
import numpy as np
import utils as hyper
import scipy.io as sio
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler, Normalizer


def hyperconvert2d(data3d):
    rows, cols, channels = data3d.shape
    data2d = data3d.reshape(rows * cols, channels, order='F')
    return data2d.transpose()


def hyperconvert3d(data2d, rows, cols, channels):
    channels, pixnum = data2d.shape
    data3d = data2d.transpose().reshape(rows, cols, channels, order='F')
    return data3d

def hypernorm(data2d, flag):
    normdata = np.zeros(data2d.shape)
    if flag == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        normdata = scaler.fit_transform(data2d.transpose()).transpose()
    elif flag == "L2_norm":
        scaler = Normalizer()
        normdata = scaler.fit_transform(data2d.transpose()).transpose()
    else:
        print("normalization wrong!")
        exit()
    return normdata


def SandiegoDataset():
    data = sio.loadmat("Sandiego.mat")
    data3d = np.array(data["Sandiego"], dtype=float)
    data3d = data3d[0:100, 0:100, :]
    remove_bands = np.hstack(
        (range(6), range(32, 35, 1), range(93, 97, 1), range(106, 113), range(152, 166), range(220, 224)))
    data3d = np.delete(data3d, remove_bands, axis=2)
    groundtruthfile = sio.loadmat("PlaneGT.mat")
    groundtruth = np.array(groundtruthfile["PlaneGT"])
    return data3d, groundtruth


def CriDataset():
    data = sio.loadmat("Cri dataset.mat")
    data3d = np.array(data["X"], dtype=float)
    groundtruth = np.array(data["mask"])
    return data3d, groundtruth


def HydiceDataset():
    data = sio.loadmat("DataTest_ori.mat")
    groundtruth_file = sio.loadmat("groundtruth.mat")
    data3d = np.array(data["DataTest_ori"], dtype=float)
    groundtruth = np.array(groundtruth_file["groundtruth"])
    return data3d, groundtruth


class Hyperloader():
    def __init__(self, args):
        if args.dataset == 'Sandiego':
            data3d, groundtruth = SandiegoDataset()
        elif args.dataset == 'Cri':
            data3d, groundtruth = CriDataset()
        elif args.dataset == 'HYDICE':
            data3d, groundtruth = HydiceDataset()
            groundtruth = groundtruth.transpose(
            )
        else:
            raise NotImplementedError
        rows, cols, bands = data3d.shape
        data2d = hyperconvert2d(data3d)
        data2d = hypernorm(data2d.transpose(), "minmax").transpose()
        data3d = hyperconvert3d(data2d, rows, cols, bands)
        self.data = torch.from_numpy(data2d).t().unsqueeze(1).float()
        self.label = torch.from_numpy(groundtruth.reshape(rows * cols))

    def __getitem__(self, idx):
        x, y = self.data[idx], self.label[idx]
        return x, y, idx

    def __len__(self):
        return len(self.data)

