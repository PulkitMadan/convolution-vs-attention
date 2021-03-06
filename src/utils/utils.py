# From
# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

import random

import numpy as np
import torch


def freeze_backbone(args, net) -> int:
    """
    Freezes weights of all layers except the final classification layer in-place.
    Return the number of remaining trainable params
    :param args: Parsed argument namespace
    :param net: torch.Module with pre-trained weights to be frozen
    :return: Number of trainable params as an integer
    """
    trainable_params = 0
    if 'convnext' in args.model:
        target = 'head'
    else:
        target = 'fc'
    for name, param in net.named_parameters():
        if target not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += param.flatten().size()[0]
    return trainable_params


# define seed for reproducibility
def seed_all(seed):
    """
    Seed Python, Numpy and Pytorch for reproducibility.
    """

    if not seed:
        seed = 123

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return rng


def seed_worker(worker_id):
    """
    Seed data loader workers
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# free up cuda memory
def freemem():
    with torch.no_grad():
        torch.cuda.empty_cache()
