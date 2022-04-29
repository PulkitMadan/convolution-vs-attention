import torch.nn as nn
from pytorch_pretrained_vit import ViT
from torchvision import models

from default_train import load_model
from models.coatnet import coatnet_0
from models.convnext import convnext_small


def define_model(args, class_size: int):
    if args.model == 'resnet':
        net = models.resnet50(pretrained=args.pretrain)
        # Set the size of each output sample to class_size
        net.fc = nn.Linear(net.fc.in_features, class_size)
    # using torch vision (v. 0.12.0+)
    elif args.model == 'vit_16_tv':
        net = models.vit_b_16(pretrained=args.pretrain)
        net.heads.head = nn.Linear(in_features=net.heads.head.in_features, out_features=class_size, bias=True)
    elif args.model == 'vit_32_tv':
        net = models.vit_b_32(pretrained=args.pretrain)
        net.heads.head = nn.Linear(in_features=net.heads.head.in_features, out_features=class_size, bias=True)
    elif args.model == 'convnext_tv':
        net = models.convnext_small(pretrained=args.pretrain)
        net.classifier[2] = nn.Linear(in_features=net.classifier[2].in_features, out_features=class_size, bias=True)
    # using local

    # TODO: check if vit_16 works
    elif args.model == 'vit_16':
        net = ViT('B_16', pretrained=args.pretrain)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=class_size, bias=True)
    elif args.model == 'vit_32':
        net = ViT('B_32', pretrained=args.pretrain)
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=class_size, bias=True)
    elif args.model == 'convnext':
        net = convnext_small(pretrained=args.pretrain, in_22k=False)
        net.head = nn.Linear(in_features=net.head.in_features, out_features=class_size, bias=True)
    elif args.model == 'coatnet':
        # no pretrained models yet
        net = coatnet_0()
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=class_size, bias=True)
    else:
        # load original resnet models here
        net = load_model(args.model)
        net.module.fc = nn.Linear(in_features=net.module.fc.in_features, out_features=class_size, bias=True)
