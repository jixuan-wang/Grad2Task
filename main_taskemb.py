# coding=utf-8
""" Multi-task meta training. """

from __future__ import absolute_import, division, print_function

import argparse
import glob
import yaml
import logging
import os
import pathlib
import random
from itertools import chain
from shutil import Error

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.autograd import grad
from traitlets.traitlets import default

from leopard_data import LeopardDataProcessor
from main_learner import Learner
from modeling_task_emb import (FIMWithBNAdapter, FIMWithBNAdapterBatch,
                               FIMWithBNAdapterBatchTrueLabels)
from utils import (MyDataParallel, ValidationAccuracies, aggregate_accuracy, loss)

import wandb
from logging_utils import get_logger
from main_meta_training import ALL_MODELS
from main_meta_training import Learner as MetaLearner
from meta_dataset import MetaDatasetProcessor, RegularDatasetProcessor, TaskDataset
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,
                          get_linear_schedule_with_warmup)
from transformers import glue_compute_metrics as compute_metrics

logger = get_logger('Taskemb Exp')

# "cola", "mnli", "mnli-mm", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli", "snli", "scitail"
TRAIN_TASK_LIST = [ "MNLI", "MRPC", "SST-2", "QQP", "QNLI", "RTE", "SNLI" ]
VAL_TASK_LIST = [ "MNLI", "MRPC", "SST-2", "QQP", "QNLI", "RTE", "SNLI" ]
TEST_TASK_LIST = ["CoLA", "SciTail"]

class ValidationAccuracies:
    def __init__(self, early_stop_steps=5):
        self.current_best_auc = 0
        self.not_improved_for = 0
        self.earyly_stop_steps = early_stop_steps

    def is_better(self, accuracies_dict):
        is_better = False

        if self.current_best_auc < accuracies_dict['same_diff_auc']:
            is_better = True
            self.not_improved_for = 0
        else:
            self.not_improved_for += 1

        return is_better

    def early_stop(self):
        return self.not_improved_for >= self.earyly_stop_steps

    def replace(self, accuracies_dict):
        self.current_best_auc = accuracies_dict['same_diff_auc']

    def get_current_best_accuracy_dict(self):
        return {"same_diff_auc": self.current_best_auc}

class TaskembLearner(Learner):
    """ Main class for training, evaluation and testing. """
    PRINT_FREQUENCY = 2
    NUM_TEST_TASKS=10
    MODEL_CLASSES = {
        'fim-bn': FIMWithBNAdapter,
        'fim-bn-batch': FIMWithBNAdapterBatch,
        'fim-bn-batch-true-labels': FIMWithBNAdapterBatchTrueLabels
    }
    CHECKPOINT_FILE_NAME = 'exp_checkpoint.pt'
    CHECKPOINT_DIR_NAME = {n:'checkpoint-'+n.upper() for n in ['current-best', 'latest', 'final']}

    def __init__(self, args):
        super().__init__(args)

        # Init optimizer
        if 'leopard' in self.args.model_type:
            # MAML based models
            self.optimizer = AdamW(self.model.meta_parameters(),
                                lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            self.optimizer = AdamW(self.model.parameters(),
                                lr=args.learning_rate, eps=args.adam_epsilon)
        self.optimizer.zero_grad()

        self.validation_accuracies = ValidationAccuracies(early_stop_steps=self.args.early_stop_patience)
        self.start_iteration = 0

    def init_model(self):
        """ Initilize model, load pretrained model and tokenizer. """
        # Load BERT config
        config = self.config_class.from_pretrained(self.bert_model_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        # Update config with adapter config
        if '-bn' in self.args.model_type:
            config.bn_adapter_hidden_size = self.args.bn_adapter_hidden_size

        config.task_emb_size = self.args.task_emb_size

        # Load tokenizer
        tokenizer = self.tokenizer_class.from_pretrained(
                        self.bert_model_name,
                        do_lower_case=True,
                        cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        # Load model
        model = self.model_class(self.args, self.bert_model_name, config)
        if self.args.bert_pretrained.lower() != 'none':
            logger.info('loading pretrained BERT model from ' + self.args.bert_pretrained)
            model.load_pretrained_encoder(torch.load(self.args.bert_pretrained)['model_state_dict'],
                                               self.args.protonet_dist_metric)
        model.to(self.device)

        return tokenizer, model

    def _load_train_data(self):
        return TaskDataset(self.args,
                            self.tokenizer,
                            TRAIN_TASK_LIST,
                            VAL_TASK_LIST,
                            TEST_TASK_LIST)

    def train(self):
        self.model.train()
        self.model.zero_grad()
        total_iterations = self.args.num_training_iterations
        train_accuracies = []
        losses = []
        sum_pi_grads = None # Accumulation of gradients of pi networks

        for iteration in range(self.start_iteration, total_iterations):
            # Sample a task
            meta_batch = self.dataset.get_train_episode_different_task_on_each_device(
                self.args.num_episodes_per_device,
                self.device_list,
                self.args.min_shots,
                self.args.max_shots,
                num_per_task=4)

            outputs = self.model(meta_batch)
            task_loss, task_accuracy = outputs[0], outputs[1]
            if self.args.n_gpu > 1:
                task_loss = sum(task_loss)
            # Whether to use the output gradients
            use_output_grads = True if len(outputs) > 2 else False
            task_pi_grads = outputs[2] if use_output_grads else None

            task_loss = task_loss # / self.args.num_episodes_per_optimize_step
            if not use_output_grads:
                # If gradients are not output, do backward()
                task_loss.backward()
            else:
                # Store the output gradients and apply them in the next optimization step
                if len(task_pi_grads) > 1:
                    task_pi_grads = [torch.add(*g) for g in zip(*task_pi_grads)]
                else:
                    task_pi_grads = task_pi_grads[0]
                if sum_pi_grads is None:
                    sum_pi_grads = task_pi_grads
                else:  # Accumulate all gradients from different episode learner
                    sum_pi_grads = [torch.add(i, j) for i, j in zip(sum_pi_grads, task_pi_grads)]

            # Store results of the current batch
            train_accuracies.append(task_accuracy.reshape(-1))
            losses.append(task_loss.item())

            # Optimize & Log
            num_episodes_so_far = (iteration + 1) * self.args.num_episodes_per_device * max(1, self.args.n_gpu)
            if ( (iteration + 1) % self.args.num_iterations_per_optimize_step == 0) or (iteration == (total_iterations - 1)):
                if not use_output_grads:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    dummy_loss = self.model.dummy_forward(meta_batch)
                    self.optimize_with_pi_gradients(dummy_loss, sum_pi_grads)

                t_loss = sum(losses)
                t_acc = torch.cat(train_accuracies).mean().item()
                results = { 'train_loss': t_loss, 'train_acc': t_acc,
                            'iteration': iteration, 'num_episodes': num_episodes_so_far }
                if self.args.wandb is not None:
                    wandb.log(results)
                # logger.info('\n'.join([f'{k}: {v}' for k, v in results.items()]))
                train_accuracies = []
                losses = []

            if (iteration + 1) % self.args.checkpoint_freq == 0:
                self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['latest'])

            # Validate every val_freq optimization steps
            if ((iteration + 1) % self.args.val_freq == 0) and (iteration != (total_iterations - 1)):
                self.model.eval()
                accuracy_dict = self.validate()
                logger.info('Validation results:')
                logger.info('\n'.join([f'{k}: {v}' for k, v in accuracy_dict.items()]))
                if self.args.wandb is not None:
                    wdict = {'eval_'+k : v for k, v in accuracy_dict.items()}
                    wdict['eval_iteration'] = iteration
                    wdict['eval_num_episodes'] = num_episodes_so_far
                    wandb.log(wdict)
                # save the model if validation is the best so far
                if self.validation_accuracies.is_better(accuracy_dict):
                    self.validation_accuracies.replace(accuracy_dict)
                    # torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                    logger.info('Best validation model was updated.')
                    self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['current-best'])
                self.model.train()

        # Save the final model
        torch.save(self.model.state_dict(), self.checkpoint_path_final)
        self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['final'])

    def validate(self):
        # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
        logger.info("***** Running evaluation *****")
        accuracy_list = []
        for batch in self.dataset.val_episode_loop_different_task_on_each_device(
                                    num_episodes_per_device=self.args.num_episodes_per_device,
                                    device_list=self.device_list):
            loss, acc = self.model(batch)
            del loss
            acc = acc.detach()
            for a in acc.reshape(-1):
                if a != -1:
                    accuracy_list.append(a.reshape(-1))
        accuracy = torch.cat(accuracy_list).mean() * 100.0
        accuracy_dict = {}
        accuracy_dict['same_diff_auc'] = accuracy

        return accuracy_dict

    def test(self):
        test_set = LeopardDataProcessor(self.args, self.tokenizer)
        if self.args.n_gpu > 1:
            logger.debug('set dataparallel for bert encoder')
            self.model.set_bert_dataparallel()
        num_shots_accuracy_dict = {}

        if len(self.args.mode.split('_')) > 2:
            test_shot_list = [int(self.args.mode.split('_')[-1])]
        else:
            test_shot_list = test_set.NUM_SHOTS_LIST

        for num_shots in test_shot_list:
            test_dir = os.path.join(self.args.output_dir, f"leopard-same-diff-{self.start_iteration}-iteration")
            pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True) 

            r_file = os.path.join(test_dir, f'K{num_shots}_results.bin')

            if os.path.exists(r_file):
                logger.info(f'Test results for {num_shots} found in {r_file}, skip testing on this.')
                continue
            
            device_list = [self.device_list[0]]
            pred_list = []
            labels_list = []
            total_num_batch = 0
            for meta_batch in test_set.episode_loop_for_taskemb(num_shots,
                                                                self.args.num_episodes_per_device,
                                                                device_list):
                total_num_batch += 1
            cur_batch = 1
            for meta_batch in test_set.episode_loop_for_taskemb(num_shots,
                                                                self.args.num_episodes_per_device,
                                                                device_list):
                logger.debug(f'batch {cur_batch} / {total_num_batch}'); cur_batch += 1
                pred, labels = self.model(meta_batch, eval=True)
                pred_list.append(pred)
                labels_list.append(labels)
            pred = torch.cat(pred_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            auc = roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy())
            logger.info(f'Test auc: {auc}')
            logger.info(f'Saving testing results to {r_file}')
            torch.save({'Test auc': auc,
                        'pred': pred,
                        'labels': labels}, r_file)
            with open(os.path.join(test_dir, f'{num_shots}_shot_slurm_id'), 'w+') as f:
                f.write(self.args.slurm_job_id)
   
    def analyze(self):
        # Init training dataset
        self.dataset = RegularDatasetProcessor(self.args,
                                                self.tokenizer,
                                                TRAIN_TASK_LIST,
                                                VAL_TASK_LIST,
                                                TEST_TASK_LIST)

        self.model.eval()
        self.model.zero_grad()
        total_iterations = self.args.num_training_iterations
        train_accuracies = []
        losses = []
        sum_pi_grads = None # Accumulation of gradients of pi networks
        best_auc = -1
        for iteration in range(self.start_iteration, total_iterations):
            # Sample a task
            meta_batch = self.dataset.get_episodes_from_different_tasks_on_each_device(
                            self.args.num_episodes_per_device,
                            self.device_list)

            outputs = self.model.extract_gradients(meta_batch)
            import ipdb; ipdb.set_trace()

def init_arg_parser():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", required=True, nargs="?",
                        choices=list(TaskembLearner.MODEL_CLASSES.keys()),
                        help="Model type.")
    parser.add_argument("--leopard_data_dir", default=None, type=str, required=True,
                        help="The dir of Leopard test datasets.")
    parser.add_argument('--exp_config', default=None, type=str, required=True,
                        help="Path to the experiment config file in json format.")

    ## Experiment parameters
    parser.add_argument("--slurm_job_id", default=None, type=str, required=False,
                        help="Slurm job id.")
    parser.add_argument("--lm_type", nargs="?", default='bert-base-uncased',
                        choices=ALL_MODELS,
                        help="Pretrained language model type.")
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
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--cnap_pretrained', default=None, type=str,
                        help="Pretrained BERT with bottleneck adapters.")

    parser.add_argument('--bert_pretrained', default=None, type=str,
                        help="Pretrained BERT with bottleneck adapters.")

    return parser

def main():
    # Arg parser initialization and parsing
    parser = init_arg_parser()
    args = parser.parse_args()
    if args.wandb.strip().lower() == 'none':
        args.wandb = None

    with open(args.exp_config, 'r') as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
        args.__dict__.update(exp_config)

    learner = TaskembLearner(args)
    learner.run()

if __name__ == "__main__":
    main()
