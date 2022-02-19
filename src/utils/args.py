# Imports
import os
import argparse
import defines
import numpy as np
from dotenv import load_dotenv


def parse_args():
	parser = argparse.ArgumentParser()

	# Required mode argument
	parser.add_argument('mode', help='High level recipe: write tensors, train, test or evaluate models.')

	# Label defining arguments
	parser.add_argument('--labels', default=defines.labels_dict,
	                    help='Dict mapping label names to their index within label tensors.')

	# Training and optimization related arguments
	parser.add_argument('--epochs', default=25, type=int,
	                    help='Number of epochs, typically passes through the entire dataset, not always well-defined.')
	parser.add_argument('--batch_size', default=32, type=int,
	                    help='Mini batch size for stochastic gradient descent algorithms.')
	parser.add_argument('--patience', default=4, type=int,
	                    help='Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.')
	parser.add_argument('--training_steps', default=80, type=int,
	                    help='Number of training batches to examine in an epoch.')
	parser.add_argument('--validation_steps', default=40, type=int,
	                    help='Number of validation batches to examine in an epoch validation.')
	parser.add_argument('--iterations', default=5, type=int,
	                    help='Generic iteration limit for hyperparameter optimization, animation, and other counts.')
	parser.add_argument('--max_parameters', default=5e6, type=int,
	                    help='Maximum number of model parameters used for hyperparameter optimization, etc.')

	# Dataset related arguments
	parser.add_argument('--valid_ratio', default=0.1, type=float,
	                    help='Rate of training tensors to save for validation must be in [0.0, 1.0].')
	parser.add_argument('--test_ratio', default=0.2, type=float,
	                    help='Rate of training tensors to save for testing [0.0, 1.0].')

	# I/O dirs
	parser.add_argument('--output_dir', default=defines.output_dir, help='Directory to write models or other data out.')
	parser.add_argument('--data_dir', default=defines.data_dir,
	                    help='Directory of tensors, must be split into test/valid/train sets')

	# Run specific arguments
	parser.add_argument('--id', default='no_id',
	                    help='Identifier for this run, user-defined string to keep experiments organized.')
	parser.add_argument('--random_seed', default=12878, type=int,
	                    help='Random seed to use throughout run.')

	# Parse, set seed, load dotenv and print args
	args = parser.parse_args()
	np.random.seed(args.random_seed)
	load_dotenv(dotenv_path=defines.dotenv_file_path)
	print('Arguments are', args)

	return args
