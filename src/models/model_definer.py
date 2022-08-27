from pytorch_pretrained_vit import ViT
from torchvision import models

from models.coatnet import coatnet_0
from models.convnext import convnext_small


def define_backbone(args):
    """
    Creates the desired model architecture based on model name of args.model
    and initialize pretrained weights if args.pretrained
    :param args: Parsed argument namespace
    :return: Initialized net as torch.Module
    """
    net = None
    if args.model == 'resnet':
        net = models.resnet50(pretrained=args.pretrained)
    elif args.model == 'vit_16_tv':
        net = models.vit_b_16(pretrained=args.pretrained)
    elif args.model == 'vit_32_tv':
        net = models.vit_b_32(pretrained=args.pretrained)
    elif args.model == 'convnext_tv':
        net = models.convnext_small(pretrained=args.pretrained)
    # using local

    # TODO: check if vit_16 works
    elif args.model == 'vit_16':
        net = ViT('B_16', pretrained=args.pretrained)
    elif args.model == 'vit_32':
        net = ViT('B_32', pretrained=args.pretrained)
    elif args.model == 'convnext':
        net = convnext_small(pretrained=args.pretrained, in_22k=False)
    elif args.model == 'coatnet':
        # no pretrained models yet
        net = coatnet_0()

    assert net is not None, f'{args.model} is not a valid model name.'
    return net
