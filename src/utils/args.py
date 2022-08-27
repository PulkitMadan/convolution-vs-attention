# Imports
import argparse
import os
from datetime import datetime

from utils import defines
from utils.utils import get_checkpoint_path, args_sanity_check


def parse_args():
    parser = argparse.ArgumentParser()
    # Run-level arguments
    parser.add_argument(
        "--do_train",
        default=False,
        action='store_true',
        help="Train the model",
    )

    parser.add_argument(
        "--do_test",
        default=False,
        action='store_true',
        help="Evaluate the model and create visualizations.",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name; available models: {resnet, vit16, vit32, convnext, coatnet}",
    )

    parser.add_argument(
        "--run_id",
        default=None,
        type=str,
        help="Provide a unique run_id. Standard should be modelname_number.",
    )

    parser.add_argument(
        "--resume_run",
        default=False,  # should be set to true when testing
        action='store_true',
        help="Resume run with run ID",
    )

    parser.add_argument(
        "--frozen",
        default=False,
        action='store_true',
        help="Freeze model parameters except for last layer",
    )

    # Trainer and Datamodule arguments
    parser.add_argument(
        "--train_loader",
        default='ood_stylized_imagenet',
        type=str,
        help="Source of the train dataloader; available sources: {imagenet, stylized_imagenet, ood_stylized_imagenet}",
    )
    parser.add_argument(
        "--val_loader",
        default='stylized_imagenet',
        type=str,
        help="Source of the val dataloader; available sources: {imagenet, stylized_imagenet, ood_stylized_imagenet}",
    )
    parser.add_argument(
        "--test_loader",
        default='stylized_imagenet',
        type=str,
        help="Source of the test dataloader; available sources: {imagenet, stylized_imagenet, ood_stylized_imagenet}",
    )

    parser.add_argument(
        '--max_epochs',
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

    parser.add_argument(
        '--pretrained',
        default=True,
        type=bool,
        help='Use pretrained models.'
    )

    # I/O dirs
    parser.add_argument('--trained_model_dir',
                        default=defines.TRAINED_MODEL_DIR,
                        help='Directory to write trained models to')

    parser.add_argument('--data_dir',
                        default=defines.DATA_DIR,
                        help='Root directory with all datasets (mela, out-of-dist SIN, in-dist SIN.)')

    parser.add_argument('--root_runs_output',
                        default=defines.RUNS_OUTPUT_DIR,
                        help='Directory to save a particular run output')

    parser.add_argument('--dot_env_path',
                        default=defines.DOT_ENV_FILE,
                        help='Path of .env file')

    parser.add_argument('--random_seed',
                        default=123,
                        type=int,
                        help='Random seed to use throughout run.')

    # TODO possibly useless args. To discuss
    parser.add_argument(
        "--mela",
        default=False,
        action='store_true',
        help="use melanoma dataset",
    )

    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Mini batch size'
    )

    # Parse, set seed and print args
    args = parser.parse_args()
    if args.run_id is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
        args.run_id = f'{args.model}-{timestamp}'
    args.run_output_dir = os.path.join(args.root_runs_output, args.run_id)
    args.checkpoint_path = get_checkpoint_path(args)
    args_sanity_check(args)
    print('Arguments are', args)
    return args
