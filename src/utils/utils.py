import glob
import os

import pytorch_lightning as pl
import torch


def get_dataloaders(args, imagenet_module: pl.LightningDataModule,
                    stylized_imagenet_module: pl.LightningDataModule,
                    ood_stylized_imagenet_module: pl.LightningDataModule) -> tuple:
    """
    Returns the correct set of train, val and dataloaders depending on the corresponding args values.
    :param args: Run arguments
    :param imagenet_module: Original ImageNet lightning DataModule
    :param stylized_imagenet_module: Stylized ImageNet lightning Datamodule
    :return: tuple(train_loader, val_loader, test_loader)
    """
    train_loader = None
    val_loader = None
    test_loader = None

    # Define train loader
    if args.train_loader == 'imagenet':
        train_loader = imagenet_module.train_dataloader()
    elif args.train_loader == 'stylized_imagenet':
        train_loader = stylized_imagenet_module.train_dataloader()
    elif args.train_loader == 'ood_stylized_imagenet':
        train_loader = ood_stylized_imagenet_module.train_dataloader()

    # Define val loader
    if args.val_loader == 'imagenet':
        val_loader = imagenet_module.val_dataloader()
    elif args.val_loader == 'stylized_imagenet':
        val_loader = stylized_imagenet_module.val_dataloader()
    elif args.train_loader == 'ood_stylized_imagenet':
        val_loader = ood_stylized_imagenet_module.val_dataloader()

    # Define test loader
    if args.test_loader == 'imagenet':
        test_loader = imagenet_module.test_dataloader()
    elif args.train_loader == 'stylized_imagenet':
        test_loader = stylized_imagenet_module.test_dataloader()
    elif args.train_loader == 'ood_stylized_imagenet':
        test_loader = ood_stylized_imagenet_module.test_dataloader()

    assert train_loader is not None, f'{args.train_loader} is not a valid DataModule'
    assert val_loader is not None, f'{args.val_loader} is not a valid DataModule'
    assert test_loader is not None, f'{args.test_loader} is not a valid DataModule'

    return train_loader, val_loader, test_loader


def args_sanity_check(args) -> None:
    assert os.path.isdir(args.trained_model_dir), f'Trained model dir not valid: {args.trained_model_dir}'
    assert os.path.isdir(args.root_runs_output), f'Runs output dir not valid: {args.root_runs_output}'
    assert os.path.isdir(args.data_dir), f'Data dir not valid: {args.data_dir}'

    if args.do_train and not args.resume_run:
        assert not os.path.isdir(
            args.run_output_dir), f'Attempting to train a new model but {args.run_output_dir} already exists. Verify run_id.'

    # If resuming, output dir & checkpoint file should exist
    if args.resume_run:
        assert os.path.isdir(
            args.run_output_dir), f'--resume flag used but {args.run_output_dir} is not a valid dir. Verify run_id.'
        assert os.path.isfile(
            args.checkpoint_path), f'--resume flag used but checkpoint has not been found in  {args.checkpoint_path}. Verify run_id.'
        assert args.do_train, f'--resume flag used should always be used in tandem with --do_train. Inconsistent run parameters.'

    # If eval with no training, output dir & checkpoint file should exist
    if not args.do_train and args.do_test:
        assert os.path.isdir(
            args.run_output_dir), f'Attempting to run eval with no training, but {args.runs_output_dir} is not a valid dir. Verify run_id.'
        assert os.path.isfile(
            args.checkpoint_path), f'Attempting to run eval with no training, but no checkpoint has been found in {args.checkpoint_path}. Verify run_id.'

    assert args.val_loader == args.test_loader, f'Val loader ({args.val_loader}) and test loader ({args.test_loader}) ' \
                                                f'should not be different. '


def get_device():
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        num_device = torch.cuda.device_count()
    else:
        num_device = 0
    print(f'device: {device}')
    print(f'# of cuda device: {num_device}')
    return device, num_device


# def get_most_recent_model_run(args) -> str:
#     dirs = glob.glob(os.path.join(args.trained_model_dir, f'{args.model_name}_*'))
#     assert dirs, f'No {args.model_name} models runs in {args.trained_model_dir}'
#     return sorted(dirs)[-1]


def get_checkpoint_path(args):
    ckpts = glob.glob(os.path.join(args.run_output_dir, args.run_id, 'checkpoints'))
    if ckpts:
        return ckpts[-1]
    else:
        return None


def freemem() -> None:
    """
    Frees torch memory cache.
    :return: None
    """
    with torch.no_grad():
        torch.cuda.empty_cache()
