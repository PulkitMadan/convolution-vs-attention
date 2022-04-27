from __future__ import print_function, division

import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
from pytorch_pretrained_vit import ViT
from sklearn.metrics import classification_report
from torchvision import models  # double naming Warning

from data.load_data import dataload, dataload_Mela, dataload_combined_datasets
from default_train import model_default_train, model_save_load, load_model, model_default_train_m
from models.coatnet import coatnet_0
from models.convnext import convnext_small
from utils import args
from utils.utils import freemem, seed_all
from visualization.visual import visualize_loss_acc, shape_bias, confusion_matrix_hm, visualize_model, eval_test

cudnn.benchmark = True

# plt.ion()   # interactive mode

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# main function
def main(args):
    # load data
    if args.mela:
        print('Loaded melanoma data')
        class_size = 2
        _, dataloaders, dataset_sizes = dataload_Mela(batch_size=args.batch_size)
    elif args.combined_data:
        print('Loading combined IN and SIN data')
        class_size = 207
        _, dataloaders, dataset_sizes = dataload_combined_datasets(batch_size=args.batch_size)
    else:
        print('Loaded SIN data')
        class_size = 207
        _, dataloaders, dataset_sizes = dataload(batch_size=args.batch_size)

    # initialize model
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
        # reduced batch size else cude memory error
        # _,dataloaders,dataset_sizes= dataload(batch_size=16)
    elif args.model == 'coatnet':
        # no pretrained models yet
        net = coatnet_0()
        net.fc = nn.Linear(in_features=net.fc.in_features, out_features=class_size, bias=True)
    else:
        # load original resnet models here
        net = load_model(args.model)
        net.module.fc = nn.Linear(in_features=net.module.fc.in_features, out_features=class_size, bias=True)

    wandb.watch(net)

    # Load model from save to scratch, if granted and exist
    model_name = args.model
    if os.name == 'nt':  # windows
        path_to_model = os.path.abspath(f'../models/trained_models/{model_name}.pth')
    else:  # linux
        home_path = os.path.expanduser('~')
        path_to_model = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/models/trained_models/{model_name}.pth'

    if os.path.exists(path_to_model) and args.load:
        print('Model loaded!')
        # Change fc layer for IN & SIN trained model 207 classes (Unfrozen)
        if args.mela:
            # net = models.resnet50(pretrained=args.pretrain)
            # Set the size of each output sample to class_size
            net.fc = nn.Linear(net.fc.in_features, 207)
            net = model_save_load(save=False, model=net, path=path_to_model)
            net.fc = nn.Linear(net.fc.in_features, class_size)
        else:
            net = model_save_load(save=False, model=net, path=path_to_model)

    # Freeze layers
    if args.pretrain and args.frozen:
        trainable_params = 0
        if 'convnext' in args.model:
            target = 'head'
        else:
            target = 'fc'
        args.model = args.model + "_frozen"

        for name, param in net.named_parameters():
            if target not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params += param.flatten().size()[0]
        print(f'{trainable_params}=')
    print(f'Training on {args.model}')
    print(net)

    # Training model
    if args.train:
        freemem()

        if args.mela:
            net, net_ls, net_as = model_default_train_m(net, dataloaders, dataset_sizes, device, epoch=args.num_epoch)
        else:
            net, net_ls, net_as = model_default_train(net, dataloaders, dataset_sizes, device, epoch=args.num_epoch)

        # save model
        model_save_load(model=net, path=path_to_model)

        # save loss acc plot
        visualize_loss_acc(net_ls, net_as, name=f'{args.model}_loss_acc_plot')

    # Visualize/test model
    else:
        if args.mela:
            # acc and classification on melanoma test set
            print(eval_test(net, dataloaders, dataset_sizes))
        else:
            # shape bias calculation
            shape_bias_dict, shape_bias_df, shape_bias_df_match = shape_bias(net, dataloaders)
            print(shape_bias_dict)
            # confusion matrix plot for shape biases
            confusion_matrix_hm(shape_bias_df['pred'], shape_bias_df['lab_shape'],
                                name=f'{args.model}_shape_bias_all_cm')
            confusion_matrix_hm(shape_bias_df['pred'], shape_bias_df['lab_texture'],
                                name=f'{args.model}_texture_bias_all_cm')
            confusion_matrix_hm(shape_bias_df_match['pred'], shape_bias_df_match['lab_shape'],
                                name=f'{args.model}_shape_bias_corr_cm')
            confusion_matrix_hm(shape_bias_df_match['pred'], shape_bias_df_match['lab_texture'],
                                name=f'{args.model}_texture_bias_corr_cm')
            print('classification report shape bias')
            print(classification_report(shape_bias_df['lab_shape'], shape_bias_df['pred']))
            print('classification report texture bias')
            print(classification_report(shape_bias_df['lab_texture'], shape_bias_df['pred']))

            # visualize sample predictions
            visualize_model(net, dataloaders, name=f'{args.model}_model_pred')

    wandb.run.finish()
    print('done!')


if __name__ == "__main__":
    # Parse arguments
    args = args.parse_args()
    rng = seed_all(args.random_seed)
    # initialize wandb project
    wandb.init(project="CNNs vs Transformers", name=args.name)

    # where the magic happens
    main(args)
