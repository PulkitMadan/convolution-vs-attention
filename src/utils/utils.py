import pytorch_lightning as pl
import torch


def get_dataloaders(args, imagenet_module: pl.LightningDataModule, stylized_imagenet_module: pl.LightningDataModule):
    train_loader = None
    val_loader = None
    test_loader = None

    # Define train loader
    if args.train_loader == 'imagenet':
        train_loader = imagenet_module.train_dataloader()
    elif args.train_loader == 'stylized_imagenet':
        train_loader = stylized_imagenet_module.train_dataloader()

    # Define val loader
    if args.val_loader == 'imagenet':
        val_loader = imagenet_module.val_dataloader()
    elif args.val_loader == 'stylized_imagenet':
        val_loader = stylized_imagenet_module.val_dataloader()

    # Define test loader
    if args.test_loader == 'imagenet':
        test_loader = imagenet_module.test_dataloader()
    elif args.train_loader == 'stylized_imagenet':
        test_loader = stylized_imagenet_module.test_dataloader()

    assert train_loader is not None, f'{args.train_loader} is not a valid DataModule'
    assert val_loader is not None, f'{args.val_loader} is not a valid DataModule'
    assert test_loader is not None, f'{args.test_loader} is not a valid DataModule'

    return train_loader, val_loader, test_loader


# free up cuda memory
def freemem():
    with torch.no_grad():
        torch.cuda.empty_cache()
