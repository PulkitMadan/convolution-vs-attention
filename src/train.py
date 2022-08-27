# from __future__ import print_function, division
import os

import dotenv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.datamodules import ImageNetDataModule, StylizedImageNetDataModule, OODStylizedImageNetDataModule
from models.lightning_model import LightningModel
from models.model_definer import define_backbone
from utils.args import parse_args
from utils.utils import freemem, get_dataloaders, get_device


def main():
    # Parse arguments
    args = parse_args()
    if os.path.isfile(args.dot_env_path):
        dotenv.load_dotenv(args.dot_env_path)
    else:
        print("Warning! No .env file found.")
    pl.seed_everything(args.random_seed, workers=True)

    if not os.path.isdir(args.run_output_dir):
        os.mkdir(args.run_output_dir)

    # Set device
    device, num_device = get_device()

    # Init data modules
    imagenet_module = ImageNetDataModule()
    stylized_imagenet_module = StylizedImageNetDataModule()
    ood_stylized_imagenet_module = OODStylizedImageNetDataModule()

    # Call setup() to init data loaders
    imagenet_module.setup()
    stylized_imagenet_module.setup()
    ood_stylized_imagenet_module.setup()

    # Get right combination of dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args, imagenet_module, stylized_imagenet_module,
                                                            ood_stylized_imagenet_module)

    # Init pretrained backbone & model
    backbone = define_backbone(args)
    model = LightningModel(backbone, num_target_class=207, freeze_backbone=args.frozen)

    if args.resume_run or (not args.do_train and args.do_test):
        print(f'Loading model from {args.checkpoint_path}')
        model.load_from_checkpoint(args.checkpoint_path)

    if args.do_train:
        freemem()
        logger = WandbLogger(project='CNNs vs Transformers', name=args.run_id)
        checkpoint_name = args.model + '-{epoch}-{val_loss:.2f}'
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=args.patience),
            ModelCheckpoint(dirpath=args.run_output_dir, filename=checkpoint_name)
        ]
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, max_epochs=args.max_epochs, devices='auto',
                             accelerator='auto', auto_scale_batch_size="binsearch", auto_lr_find=True)
        trainer.tune(model)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # --------- OLD ---------------

    #
    # # Training model
    # if args.train:
    #     freemem()
    #
    #     if args.mela:
    #         net, net_ls, net_as = model_default_train_m(net, args, dataloaders, dataset_sizes, device,
    #                                                     epoch=args.num_epoch)
    #     else:
    #         net, net_ls, net_as = model_default_train(net, args, dataloaders, dataset_sizes, device,
    #                                                   epoch=args.num_epoch)
    #
    #     # save model
    #     model_save_load(model=net, path=path_to_model)
    #
    #     # save loss acc plot
    #     visualize_loss_acc(net_ls, net_as, name=f'{args.model}_loss_acc_plot')
    #
    # # Visualize/test model
    # else:
    #     if args.mela:
    #         # acc and classification on melanoma test set
    #         print(eval_test(net, dataloaders, dataset_sizes))
    #     else:
    #         # shape bias calculation
    #         shape_bias_dict, shape_bias_df, shape_bias_df_match = shape_bias(net, dataloaders)
    #         print(shape_bias_dict)
    #         # confusion matrix plot for shape biases
    #         confusion_matrix_hm(shape_bias_df['pred'], shape_bias_df['lab_shape'],
    #                             name=f'{args.model}_shape_bias_all_cm')
    #         confusion_matrix_hm(shape_bias_df['pred'], shape_bias_df['lab_texture'],
    #                             name=f'{args.model}_texture_bias_all_cm')
    #         confusion_matrix_hm(shape_bias_df_match['pred'], shape_bias_df_match['lab_shape'],
    #                             name=f'{args.model}_shape_bias_corr_cm')
    #         confusion_matrix_hm(shape_bias_df_match['pred'], shape_bias_df_match['lab_texture'],
    #                             name=f'{args.model}_texture_bias_corr_cm')
    #         print('classification report shape bias')
    #         print(classification_report(shape_bias_df['lab_shape'], shape_bias_df['pred']))
    #         print('classification report texture bias')
    #         print(classification_report(shape_bias_df['lab_texture'], shape_bias_df['pred']))
    #
    #         # visualize sample predictions
    #         visualize_model(net, dataloaders, name=f'{args.model}_model_pred')
    #
    # wandb.run.finish()
    # print('done!')


if __name__ == "__main__":
    # where the magic happens
    main()
