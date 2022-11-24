""" Multi-task meta training. """

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import pathlib
import random
from shutil import Error

import numpy as np
import torch

from leopard_data import LeopardDataProcessor
from utils import MyDataParallel

import wandb
from logging_utils import get_logger
from meta_dataset import MetaDatasetProcessor
from transformers import (BertConfig, BertTokenizer)

logger = get_logger('Meta-Training')

# "cola", "mnli", "mnli-mm", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli", "snli", "scitail"
TRAIN_TASK_LIST = [ "MNLI", "MRPC", "SST-2", "QQP", "QNLI", "RTE", "SNLI" ]
VAL_TASK_LIST = [ "MNLI", "MRPC", "SST-2", "QQP", "QNLI", "RTE", "SNLI" ]
# TRAIN_TASK_LIST = [ "RTE"]
# VAL_TASK_LIST = [ "RTE"]
TEST_TASK_LIST = ["CoLA", "SciTail"]

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

class Engine(object):
    """ Main class for training, evaluation and testing. """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Initial setup
        self.parallel_episode_per_device = True
        self.device, self.device_list = self.setup_device()
        self.setup()

        # Init model
        self.bert_model_name = 'bert-base-uncased'
        self.config_class, self.tokenizer_class = BertConfig, BertTokenizer
        self.model_class = self.MODEL_CLASSES[args.model_type]
        self.tokenizer, self.model = self.init_model()
        self.start_iteration = 0

    def set_seed(self):
        args = self.args
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def setup(self):
        args = self.args
        self.set_seed()  # Added here for reproductibility (even between python 2 and 3)

        args.num_episodes_per_optimize_step = max(1, len(self.device_list)) * args.num_episodes_per_device * args.num_iterations_per_optimize_step

        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        # Save args into output directory
        with open(os.path.join(args.output_dir, 'run_args.txt'), 'w') as f:
            f.write(json.dumps(args.__dict__, indent=2))

        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
            and 'train' in args.mode and not args.overwrite_output_dir:
            raise ValueError(
                '''Output directory ({}) already exists and is not empty. Use
                    --overwrite_output_dir to overcome.'''.format(args.output_dir))

        # Init wandb
        if 'train' in args.mode and args.wandb is not None:
            wandb.init(project=args.wandb, config=vars(args))
            wandb.config.update(args)

    def setup_device(self):
        args = self.args
        device_list = None
        # Setup CUDA, GPU & distributed training
        # if args.local_rank == -1 or args.no_cuda:
        args.n_gpu = torch.cuda.device_count()
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if args.n_gpu > 1 and not 'leopard' in self.args.model_type:
            device_list = [torch.device(f"cuda:{i}") for i in range(args.n_gpu)]
        else:
            device_list = [device]
        # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        #     torch.cuda.set_device(args.local_rank)
        #     device = torch.device("cuda", args.local_rank)
        #     torch.distributed.init_process_group(backend='nccl')
        #     args.n_gpu = 1
        return device, device_list

    def setup_dataparallel(self):
        if 'leopard' in self.args.model_type:
            # We use a different parallel training for Leopard: parallel forward/backward 
            # on multi gpus for each episode, while for other modes, we run multiple 
            # episodes on gpus in parallel.
            if self.args.n_gpu > 1:
                self.model.setup_dataparallel()
        else:
            if self.args.n_gpu > 1:
                self.model = MyDataParallel(self.model)

    def run(self):
        if self.args.mode == 'train':
            self.run_train()
        if 'test' in self.args.mode:
            self.run_test()
        if 'analyze' in self.args.mode:
            self.run_analyze()
        
    def run_train(self):
        # Load and prepare model
        if not self.args.start_from_scratch:
            if self.args.checkpoint_path is not None:
                self.resume_from_checkpoint(self.args.checkpoint_path)
            else:
                self.resume_from_latest_or_best(load_from_latest=True, exit_if_failed=False)
        self.setup_dataparallel()

        # Init training dataset
        self.dataset = self._load_train_data()

        self.train()
    
    def _load_train_data(self):
        return MetaDatasetProcessor(self.args,
                                    self.tokenizer,
                                    TRAIN_TASK_LIST,
                                    VAL_TASK_LIST,
                                    TEST_TASK_LIST)
        
        
    def train(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()
    
    def save_checkpoint(self, iteration, name='default'):
        checkpoint_output_dir = os.path.join(self.args.output_dir, name)
        if not os.path.exists(checkpoint_output_dir):
            os.makedirs(checkpoint_output_dir)

        temp_path = os.path.join(checkpoint_output_dir, 'temp.pt')
        torch.save({
            'iteration': iteration,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.validation_accuracies.get_current_best_accuracy_dict(),
            'model_state_dict': self.model.module.state_dict() if type(self.model) is MyDataParallel else self.model.state_dict(),
            'training_args': self.args,
            'rng' : torch.random.get_rng_state(),
            'np_rand_state': np.random.get_state()
        }, temp_path)
        os.replace(temp_path, os.path.join(checkpoint_output_dir, self.CHECKPOINT_FILE_NAME))
        # self.tokenizer.save_pretrained(self.args.checkpoint_output_dir)
        logger.info(f"Saved iteration {iteration} to {checkpoint_output_dir}")
        #TODO: Save and load early stop steps.

    def resume_from_checkpoint(self, checkpoint_path=None, model_only=False):
        if os.path.exists(os.path.join(checkpoint_path, self.CHECKPOINT_FILE_NAME)):
            logger.info(f'Loading from given checkpoint path: {checkpoint_path}')
            self.load_checkpoint(checkpoint_path, model_only)
        else:
            raise Error(f'Was asked to load from {checkpoint_path} but cound not find checkpoint file in it.')

    def resume_from_latest_or_best(self, load_from_latest=True, exit_if_failed=False):
        ckp_path = os.path.join(self.args.output_dir,
                                self.CHECKPOINT_DIR_NAME['current-best'] if not load_from_latest else self.CHECKPOINT_DIR_NAME['latest'])
        if os.path.exists(os.path.join(ckp_path, self.CHECKPOINT_FILE_NAME)):
            logger.info(f'Loading from checkpoint path: {ckp_path}')
            self.load_checkpoint(ckp_path)
        else:
            if not exit_if_failed:
                logger.warning(f'No checkpoint path is given nor found under {ckp_path}. Keep using the initial model.')
            else:
                raise Error(f'No checkpoint path is given nor found under {ckp_path}.')

    def load_checkpoint(self, checkpoint_path, model_only=False):
        checkpoint = torch.load(os.path.join(checkpoint_path, self.CHECKPOINT_FILE_NAME))
        if not model_only:
            self.start_iteration = checkpoint['iteration']
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                logger.warning('Failed to load optimizer from the given checkpoint, skip loading...')
            self.validation_accuracies.replace(checkpoint['best_accuracy'])
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            torch.random.set_rng_state(checkpoint['rng'])
            np.random.set_state(checkpoint['np_rand_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f"Loaded iteration {self.start_iteration} from {checkpoint_path}")

def init_arg_parser(learner):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", required=True, nargs="?",
                        choices=list(learner.MODEL_CLASSES.keys()),
                        help="Model type.")
    parser.add_argument("--leopard_data_dir", default=None, type=str, required=True,
                        help="The dir of Leopard test datasets.")


    ## Experiment parameters
    parser.add_argument("--slurm_job_id", default=None, type=str, required=False,
                        help="Slurm job id.")
    parser.add_argument("--exp_id", default=None, type=str, required=False,
                        help="Exp id.")
    parser.add_argument("--lm_type", nargs="?", default='bert-base-uncased',
                        choices=ALL_MODELS,
                        help="Pretrained language model type.")
    parser.add_argument("--checkpoint_freq", type=int, default=1000,
                        help="Number of iterations between validations.")
    parser.add_argument("--start_from_scratch", action='store_true',
                        help="Training from scratch.")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="Path to checkpoint to start from.")
    parser.add_argument("--load_from_current_best", action='store_true',
                        help="Loading from previous best checkpoint.")
    parser.add_argument("--wandb", default=None, type=str,
                        help="Project name on wandb.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--mode", default="train", type=str,
                        help="Whether do training / testing.")
                        # choices=['train', 'test_latest', 'test_best'],
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--val_freq", type=int, default=1000,
                        help="Number of iterations between validations.")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help="Local rank for distributed training.")

    # Modeling parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # Hyperparameters
    parser.add_argument("--num_shots_support", default=10, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_shots_query", default=10, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--num_episodes_per_device", default=1, type=int,
                        help="Number of parallel tasks per GPU/CPU during training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_iterations", type=int, default=110000,
                        help="Number of meta-training iterations.")
    parser.add_argument("--num_training_epochs", type=int, default=5,
                        help="Number of meta-training epochs.")
    parser.add_argument("--num_support_batches", default=8, type=int,
                        help="Number of support batches in each episode.")
    parser.add_argument("--num_query_batches", default=1, type=int,
                        help="Number of query batches in each episode.")
    parser.add_argument("--num_iterations_per_optimize_step", default=5, type=int,
                        help="Number of tasks between parameter optimizations.")
    parser.add_argument("--max_num_val_episodes", default=100, type=int,
                        help="Number of episode for evaluation of each dataset.")
    parser.add_argument("--warmup_steps_ratio", default=0, type=float,
                        help="Linear warmup over warmup_steps_ratio*total_steps.")
    parser.add_argument("--bert_linear_size", default=768, type=int,
                        help="Size of the linear layer after BERT encoder.")
    parser.add_argument("--adapter_type", default="film", nargs="?", const="film",
                        choices=['film', 'bn'],
                        help="Which type of adapter to use.")
    parser.add_argument("--protonet_dist_metric", default="euc", nargs="?", const="euc",
                        choices=['cos', 'euc'], help="Distance metric used trained ProtoNet.")
    parser.add_argument("--bn_adapter_hidden_size", default=16, type=int,
                        help="Hidden size of the bottleneck adapter network.")
    parser.add_argument("--task_emb_size", default=100, type=int,
                        help="Size of per-layer task embedding.")
    parser.add_argument("--bn_context_size", default=100, type=int,
                        help="Size of context vector.")
    parser.add_argument("--classifier_hidden_size", default=200, type=int,
                        help="Size of hidden unit in classifier.")
    parser.add_argument("--cnap_freeze_base_model", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Freeze the base model of CNAP, only train the adaptation network.")
    parser.add_argument('--cnap_adapt', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether adapting the base model using an adaptation network.")
    parser.add_argument('--cnap_pretrained', default=None, type=str,
                        help="Pretrained BERT with bottleneck adapters.")

    parser.add_argument("--early_stop_by", default="avg", nargs="?", const="avg",
                        choices=['vote', 'avg'], help="How to eatly stop.")
    parser.add_argument("--early_stop_patience", default=5, type=int,
                        help="Early stop patience.")
                    
    parser.add_argument("--fine_tune_epochs", default=10, type=int, required=False,
                        help="Fine tuning epochs for simple baseline.")
    return parser

def main():
    # Arg parser initialization and parsing
    parser = init_arg_parser()
    args = parser.parse_args()
    if args.wandb.strip().lower() == 'none':
        args.wandb = None

    learner = Learner(args)
    learner.run()

if __name__ == "__main__":
    main()
