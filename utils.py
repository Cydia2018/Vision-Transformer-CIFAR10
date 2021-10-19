""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_network(args):
    """ return given network
    """

    if args.net == 'vit':
        from vit_pytorch.vit import ViT
        net = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    # elif args.net == 'deit':
    #     from vit_pytorch.distill import DistillWrapper
    #     net = vgg13_bn()
    elif args.net == 'deepvit':
        from vit_pytorch.deepvit import DeepViT
        net = DeepViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif args.net == 'cait':
        from vit_pytorch.cait import CaiT
        net = CaiT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 4,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05)
    elif args.net == 'cpvt':
        from vit_pytorch.cpvt import CPVT
        net = CPVT()
    elif args.net == 'cvt':
        from vit_pytorch.cvt import CvT
        net = CvT(
            num_classes = 10,
            s1_emb_dim = 64,        # stage 1 - dimension
            s1_emb_kernel = 7,      # stage 1 - conv kernel
            s1_emb_stride = 4,      # stage 1 - conv stride
            s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
            s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
            s1_heads = 1,           # stage 1 - heads
            s1_depth = 1,           # stage 1 - depth
            s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
            s2_emb_dim = 192,       # stage 2 - (same as above)
            s2_emb_kernel = 3,
            s2_emb_stride = 2,
            s2_proj_kernel = 3,
            s2_kv_proj_stride = 2,
            s2_heads = 3,
            s2_depth = 2,
            s2_mlp_mult = 4,
            s3_emb_dim = 384,       # stage 3 - (same as above)
            s3_emb_kernel = 3,
            s3_emb_stride = 2,
            s3_proj_kernel = 3,
            s3_kv_proj_stride = 2,
            s3_heads = 4,
            s3_depth = 10,
            s3_mlp_mult = 4,
            dropout = 0.
        )
    elif args.net == 'ceit':
        from vit_pytorch.ceit import CeiT
        net = CeiT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 256,
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dim_head = 32,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif args.net == 'levit':
        from vit_pytorch.levit import LeViT
        net = LeViT(
            image_size = 32,
            num_classes = 10,
            stages = 3,             # number of stages
            dim = (128, 192, 256),  # dimensions at each stage
            depth = 4,              # transformer of depth 4 at each stage
            heads = (4, 6, 8),      # heads at each stage
            mlp_mult = 2,
            dropout = 0.1
        )
    # elif args.net == 'googlenet':
    #     from vit_pytorch.googlenet import googlenet
    #     net = googlenet()
    # elif args.net == 'inceptionv3':
    #     from vit_pytorch.inceptionv3 import inceptionv3
    #     net = inceptionv3()
    # elif args.net == 'inceptionv4':
    #     from vit_pytorch.inceptionv4 import inceptionv4
    #     net = inceptionv4()
    # elif args.net == 'inceptionresnetv2':
    #     from vit_pytorch.inceptionv4 import inception_resnet_v2
    #     net = inception_resnet_v2()
    # elif args.net == 'xception':
    #     from vit_pytorch.xception import xception
    #     net = xception()
    # elif args.net == 'resnet18':
    #     from vit_pytorch.resnet import resnet18
    #     net = resnet18()
    # elif args.net == 'resnet34':
    #     from vit_pytorch.resnet import resnet34
    #     net = resnet34()
    # elif args.net == 'resnet50':
    #     from vit_pytorch.resnet import resnet50
    #     net = resnet50()
    # elif args.net == 'resnet101':
    #     from vit_pytorch.resnet import resnet101
    #     net = resnet101()
    # elif args.net == 'resnet152':
    #     from vit_pytorch.resnet import resnet152
    #     net = resnet152()
    # elif args.net == 'preactresnet18':
    #     from vit_pytorch.preactresnet import preactresnet18
    #     net = preactresnet18()
    # elif args.net == 'preactresnet34':
    #     from vit_pytorch.preactresnet import preactresnet34
    #     net = preactresnet34()
    # elif args.net == 'preactresnet50':
    #     from vit_pytorch.preactresnet import preactresnet50
    #     net = preactresnet50()
    # elif args.net == 'preactresnet101':
    #     from vit_pytorch.preactresnet import preactresnet101
    #     net = preactresnet101()
    # elif args.net == 'preactresnet152':
    #     from vit_pytorch.preactresnet import preactresnet152
    #     net = preactresnet152()
    # elif args.net == 'resnext50':
    #     from vit_pytorch.resnext import resnext50
    #     net = resnext50()
    # elif args.net == 'resnext101':
    #     from vit_pytorch.resnext import resnext101
    #     net = resnext101()
    # elif args.net == 'resnext152':
    #     from vit_pytorch.resnext import resnext152
    #     net = resnext152()
    # elif args.net == 'shufflenet':
    #     from vit_pytorch.shufflenet import shufflenet
    #     net = shufflenet()
    # elif args.net == 'shufflenetv2':
    #     from vit_pytorch.shufflenetv2 import shufflenetv2
    #     net = shufflenetv2()
    # elif args.net == 'squeezenet':
    #     from vit_pytorch.squeezenet import squeezenet
    #     net = squeezenet()
    # elif args.net == 'mobilenet':
    #     from vit_pytorch.mobilenet import mobilenet
    #     net = mobilenet()
    # elif args.net == 'mobilenetv2':
    #     from vit_pytorch.mobilenetv2 import mobilenetv2
    #     net = mobilenetv2()
    # elif args.net == 'nasnet':
    #     from vit_pytorch.nasnet import nasnet
    #     net = nasnet()
    # elif args.net == 'attention56':
    #     from vit_pytorch.attention import attention56
    #     net = attention56()
    # elif args.net == 'attention92':
    #     from vit_pytorch.attention import attention92
    #     net = attention92()
    # elif args.net == 'seresnet18':
    #     from vit_pytorch.senet import seresnet18
    #     net = seresnet18()
    # elif args.net == 'seresnet34':
    #     from vit_pytorch.senet import seresnet34
    #     net = seresnet34()
    # elif args.net == 'seresnet50':
    #     from vit_pytorch.senet import seresnet50
    #     net = seresnet50()
    # elif args.net == 'seresnet101':
    #     from vit_pytorch.senet import seresnet101
    #     net = seresnet101()
    # elif args.net == 'seresnet152':
    #     from vit_pytorch.senet import seresnet152
    #     net = seresnet152()
    # elif args.net == 'wideresnet':
    #     from vit_pytorch.wideresidual import wideresnet
    #     net = wideresnet()
    # elif args.net == 'stochasticdepth18':
    #     from vit_pytorch.stochasticdepth import stochastic_depth_resnet18
    #     net = stochastic_depth_resnet18()
    # elif args.net == 'stochasticdepth34':
    #     from vit_pytorch.stochasticdepth import stochastic_depth_resnet34
    #     net = stochastic_depth_resnet34()
    # elif args.net == 'stochasticdepth50':
    #     from vit_pytorch.stochasticdepth import stochastic_depth_resnet50
    #     net = stochastic_depth_resnet50()
    # elif args.net == 'stochasticdepth101':
    #     from vit_pytorch.stochasticdepth import stochastic_depth_resnet101
    #     net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    cifar10_training_loader = DataLoader(
        cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=False):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

def compute_mean_std(cifar10_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar10_dataset[i][1][:, :, 0] for i in range(len(cifar10_dataset))])
    data_g = numpy.dstack([cifar10_dataset[i][1][:, :, 1] for i in range(len(cifar10_dataset))])
    data_b = numpy.dstack([cifar10_dataset[i][1][:, :, 2] for i in range(len(cifar10_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]