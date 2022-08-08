from __future__ import print_function, division

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import classification_report

from data.datamodules import ImageNetDataModule, StylizedImageNetDataModule
from data.load_data import dataload, dataload_Mela, dataload_combined_datasets
from default_train import model_default_train, model_save_load, model_default_train_m
from models.model_definer import define_backbone
from utils.args import parse_args
from utils.utils import freemem, get_dataloaders
from visualization.visual import visualize_loss_acc, shape_bias, confusion_matrix_hm, visualize_model, eval_test

torch.backends.cudnn = True


# main function
def main(args):
    imagenet_module = ImageNetDataModule()
    stylized_imagenet_module = StylizedImageNetDataModule()

    # Initialize the datamodules to have direct access to dataloaders
    imagenet_module.setup()
    stylized_imagenet_module.setup()

    train_loader, val_loader, test_loader = get_dataloaders(args, imagenet_module, stylized_imagenet_module)

    # --------- OLD ---------------
    # load data
    if args.mela:
        class_size = 2
        _, dataloaders, dataset_sizes = dataload_Mela(args, batch_size=args.batch_size)
        print('Loaded melanoma data')
    elif args.combined_data:
        class_size = 207
        _, dataloaders, dataset_sizes = dataload_combined_datasets(args, batch_size=args.batch_size)
        print('Loading combined IN and SIN data')
    else:
        class_size = 207
        _, dataloaders, dataset_sizes = dataload(args, batch_size=args.batch_size)
        print('Loaded SIN data')

    # initialize model
    net = define_backbone(args, class_size)
    wandb.watch(net)

    # Load model from save to scratch, if granted and exist
    model_name = args.model

    # if os.name == 'nt':  # windows
    #     path_to_model = os.path.abspath(f'../models/trained_models/{model_name}.pth')
    # else:  # linux
    #     home_path = os.path.expanduser('~')
    #     path_to_model = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/models/trained_models/{model_name}.pth'

    path_to_model = os.path.join(args.trained_model_dir, f'{model_name}.pth')
    if os.path.exists(path_to_model) and args.load:
        print('Model loaded!')
        # Change fc layer for IN & SIN trained model 207 classes (Unfrozen)
        if args.mela:
            net.fc = nn.Linear(net.fc.in_features, 207)
            net = model_save_load(save=False, model=net, path=path_to_model)
            net.fc = nn.Linear(net.fc.in_features, class_size)
        else:
            net = model_save_load(save=False, model=net, path=path_to_model)

    # Freeze pre-trained layers
    if args.pretrain and args.frozen:
        args.model = args.model + "_frozen"
        trainable_params = freeze_backbone(args, net)
        print(f'{trainable_params}=')

    print(f'Training on {args.model}')
    print(net)

    # Training model
    if args.train:
        freemem()

        if args.mela:
            net, net_ls, net_as = model_default_train_m(net, args, dataloaders, dataset_sizes, device,
                                                        epoch=args.num_epoch)
        else:
            net, net_ls, net_as = model_default_train(net, args, dataloaders, dataset_sizes, device,
                                                      epoch=args.num_epoch)

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
    args = parse_args()
    rng = pl.seed_everything(args.random_seed, workers=True)

    # Set device
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    if torch.cuda.is_available():
        num_device = torch.cuda.device_count()
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))

    # initialize wandb project
    wandb.init(project="CNNs vs Transformers", name=args.name)

    # where the magic happens
    main(args)
