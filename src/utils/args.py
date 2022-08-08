# Imports
import argparse
import os.path

from utils import defines


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=False,
        action='store_true',  # 'store_false' if want default to false
        help="Train or Test",
    )
    parser.add_argument(
        "--model",
        default='resnet',
        type=str,
        help="Model name; available models: {resnet, vit_16, vit_32, convnext, coatnet}",
    )
    parser.add_argument(
        "--pretrain",
        default=False,
        action='store_true',
        help="Load pretrained model parameters",
    )
    parser.add_argument(
        "--load",
        default=False,  # should be set to true when testing
        action='store_true',
        help="Load saved model parameters",
    )

    parser.add_argument(
        "--frozen",
        default=False,
        action='store_true',
        help="Freeze model parameters except for last layer",
    )

    parser.add_argument(
        "--train_loader",
        default='imagenet',
        type=str,
        help="Source of the train dataloader; available sources: {imagenet, stylized_imagenet}",
    )
    parser.add_argument(
        "--val_loader",
        default='imagenet',
        type=str,
        help="Source of the val dataloader; available sources: {imagenet, stylized_imagenet}",
    )
    parser.add_argument(
        "--test_loader",
        default='imagenet',
        type=str,
        help="Source of the test dataloader; available sources: {imagenet, stylized_imagenet}",
    )

    parser.add_argument(
        "--mela",
        default=False,
        action='store_true',
        help="use melanoma dataset",
    )



    parser.add_argument(
        "--name",
        type=str,
        help="Name of the run",
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Mini batch size'
    )

    parser.add_argument(
        '--num_epoch',
        default=80,
        type=int,
        help='Number of epoch to train. Geirhos uses 80.'
    )

    parser.add_argument(
        '--patience',
        default=5,
        type=int,
        help='Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.'
    )

    # I/O dirs
    parser.add_argument('--trained_model_dir',
                        default=defines.TRAINED_MODEL_DIR,
                        help='Directory to write trained models to')

    parser.add_argument('--data_dir',
                        default=defines.DATA_DIR,
                        help='Root directory with all datasets (mela, out-of-dist SIN, in-dist SIN.)')

    parser.add_argument('--random_seed',
                        default=123,
                        type=int,
                        help='Random seed to use throughout run.')

    # Parse, set seed and print args
    args = parser.parse_args()
    assert os.path.isdir(args.trained_model_dir), f'Trained model dir not valid: {args.trained_model_dir}'
    assert os.path.isdir(args.data_dir), f'data dir not valid: {args.trained_model_dir}'
    print('Arguments are', args)

    return args
