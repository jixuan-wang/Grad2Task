# coding=utf-8
""" Multi-task meta training. """

from __future__ import absolute_import, division, print_function

import argparse
import gc
import glob
import json
import logging
import os
import pathlib
import random
from itertools import chain
from shutil import Error

import numpy as np
import torch
from torch.autograd import grad
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              Subset, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from traitlets.traitlets import default

from leopard_data import LeopardDataProcessor, LeopardDataProcessor_0526
from modeling_baselines import FineTuningBaseline
from utils import (MyDataParallel, ValidationAccuracies,
                   ValidationAccuraciesByAverage, aggregate_accuracy, loss)

import wandb
from logging_utils import get_logger
from engine import Engine
from meta_dataset import MetaDatasetProcessor
from modeling_cnap import (CNAPCLSShiftScaleARPretrainedTaskEmb,
                           CNAPWithBNAdapter,
                           CNAPWithBNAdapter_AdaptContextShiftScaleAR,
                           CNAPWithBNAdapterAdaptContextCLSShiftScale,
                           CNAPWithBNAdapterAdaptContextCLSShiftScaleAR,
                           CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_Shared,
                           CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_X,
                           CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_X_Grad,
                           CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_XY,
                           HyperWithBNAdapter,
                           HyperWithBNAdapterPretrainedTaskEmb)
from modeling_leopard import LeopardFixedLR, LeopardForMetatraining
# from modeling_bert_adaptation import (BertWithFilmAdapterForMetatraining)
from modeling_protonet import (
    ProtoNetForTextClassification, ProtoNetForTextClassificationEuclidean,
    ProtoNetForTextClassificationWithBNAdapter,
    ProtoNetForTextClassificationWithBNAdapterEuclidean,
    ProtoNetForTextClassificationWithLayerNormAdaptation,
    ProtoNetForTextClassificationWithLayerNormAdaptationEuclidean)
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,
                          get_linear_schedule_with_warmup)
from transformers import glue_compute_metrics as compute_metrics

logger = get_logger('Meta-Training')

# "cola", "mnli", "mnli-mm", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli", "snli", "scitail"
TRAIN_TASK_LIST = [ "MNLI", "MRPC", "SST-2", "QQP", "QNLI", "RTE", "SNLI" ]
VAL_TASK_LIST = [ "MNLI", "MRPC", "SST-2", "QQP", "QNLI", "RTE", "SNLI" ]
# TRAIN_TASK_LIST = [ "RTE"]
# VAL_TASK_LIST = [ "RTE"]
TEST_TASK_LIST = ["CoLA", "SciTail"]

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, )), ())

class MetaLearner(Engine):
    """ Main class for training, evaluation and testing. """
    PRINT_FREQUENCY = 1000
    NUM_TEST_TASKS=10
    MODEL_CLASSES = {
        ### Baselines
        'fine-tune-baseline': FineTuningBaseline,
        ### Leopard
        'bert-leopard': LeopardForMetatraining,
        'bert-leopard-fixlr': LeopardFixedLR,
        ### ProtoNet
        'bert-protonet': ProtoNetForTextClassification,
        'bert-protonet-bn': ProtoNetForTextClassificationWithBNAdapter,
        'bert-protonet-layer-norm': ProtoNetForTextClassificationWithLayerNormAdaptation,
        'bert-protonet-euc': ProtoNetForTextClassificationEuclidean,
        'bert-protonet-euc-bn': ProtoNetForTextClassificationWithBNAdapterEuclidean,
        'bert-protonet-euc-layer-norm': ProtoNetForTextClassificationWithLayerNormAdaptationEuclidean,
        ### CNAP
        'bert-cnap-bn': CNAPWithBNAdapter,
        # 'bert-cnap-bn-euc-ar-fim': CNAPWithBNAdapterAdaptInputOutput_AR_FIM,
        # 'bert-cnap-bn-euc-ar-fim-new': CNAPWithBNAdapterAdaptInputOutput_AR_FIM_New,
        # 'bert-cnap-bn-euc-context': CNAPWithBNAdapterAdaptContext,
        # 'bert-cnap-bn-euc-context-cls': CNAPWithBNAdapterAdaptContextCLS,
        'bert-cnap-bn-euc-context-cls-shift-scale': CNAPWithBNAdapterAdaptContextCLSShiftScale,
        # Current best
        'bert-cnap-bn-euc-context-cls-shift-scale-ar': CNAPWithBNAdapterAdaptContextCLSShiftScaleAR,
        'bert-cnap-bn-euc-context-shift-scale-ar': CNAPWithBNAdapter_AdaptContextShiftScaleAR,
        'bert-cnap-bn-euc-context-cls-shift-scale-ar-X': CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_X,
        'bert-cnap-bn-euc-context-cls-shift-scale-ar-XGrad': CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_X_Grad,
        'bert-cnap-bn-euc-context-cls-shift-scale-ar-XY': CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_XY,
        'bert-cnap-bn-euc-context-cls-shift-scale-ar-shared': CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_Shared,
        # 'bert-cnap-bn-euc-context-cls-shift-scale-ar-larger': CNAPWithBNAdapterAdaptContextCLSShiftScaleARLarger,
        # 'bert-cnap-bn-euc-context-cls-mlp-shift-scale-ar': CNAPWithBNAdapterAdaptContextCLSMLPTaskEmbShiftScaleAR,
        # 'bert-cnap-bn-euc-context-cls-shift-scale-ar-lnorm': CNAPWithBNAdapterAdaptContextCLSShiftScaleARLNorm,
        'bert-cnap-bn-pretrained-taskemb': CNAPCLSShiftScaleARPretrainedTaskEmb,
        # 'bert-cnap-bn-euc-cls-out': CNAPWithBNAdapterCLSAROut,
        ### Hypternet
        'bert-cnap-bn-hyper': HyperWithBNAdapter,
        'bert-cnap-bn-hyper-pretrained-taskemb': HyperWithBNAdapterPretrainedTaskEmb
    }
    CHECKPOINT_FILE_NAME = 'exp_checkpoint.pt'
    CHECKPOINT_DIR_NAME = {n:'checkpoint-'+n.upper() for n in ['current-best', 'latest', 'final']}

    def __init__(self, args):
        super().__init__(args)
        if 'train' in self.args.mode:
            # Init optimizer
            if 'leopard' in self.args.model_type:
                # MAML based models
                self.optimizer = AdamW(self.model.meta_parameters(),
                                    lr=args.learning_rate, eps=args.adam_epsilon)
            else:
                self.optimizer = AdamW(self.model.parameters(),
                                    lr=args.learning_rate, eps=args.adam_epsilon)
            self.optimizer.zero_grad()

        if self.args.early_stop_by == 'avg':
            self.validation_accuracies = ValidationAccuraciesByAverage(VAL_TASK_LIST,
                                                                       early_stop_steps=self.args.early_stop_patience)
            logger.warning('Early stopping by average performance')
        elif self.args.early_stop_by == 'vote':
            self.validation_accuracies = ValidationAccuracies(VAL_TASK_LIST,
                                                              early_stop_steps=self.args.early_stop_patience)
            logger.warning('Early stopping by voting')
        self.start_iteration = 0

    def init_model(self):
        """ Initilize model, load pretrained model and tokenizer. """
        # Load BERT config
        config = self.config_class.from_pretrained(self.bert_model_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        # Update config with adapter config
        if '-bn' in self.args.model_type:
            config.bn_adapter_hidden_size = self.args.bn_adapter_hidden_size

        if 'cnap' in self.args.model_type:
            config.task_emb_size = self.args.task_emb_size

        # Load tokenizer
        tokenizer = self.tokenizer_class.from_pretrained(self.bert_model_name,
            do_lower_case=True,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None)

        # Load model
        if self.args.cnap_pretrained.lower() != 'none':
            assert 'cnap' in self.args.model_type
            logger.info('loading pretrained bert-protonet-bn model from ' + self.args.cnap_pretrained)
            model = self.model_class(self.args, self.bert_model_name, config,
                                     pt_encoder_dist_metric=self.args.protonet_dist_metric,
                                     pt_encoder_state_dict=torch.load(self.args.cnap_pretrained)['model_state_dict'])
        else:
            model = self.model_class(self.args, self.bert_model_name, config)
                                     # pt_encoder_dist_metric=self.args.protonet_dist_metric,
                                     # pt_encoder_state_dict=None)
        model.to(self.device)
        
        # if 'train' in self.args.mode and self.args.wandb is not None:
        #     wandb.watch(model, log="all")
        return tokenizer, model

    def train(self):
        self.model.train()
        self.model.zero_grad()
        # total_iterations = self.args.num_training_iterations
        train_accuracies = []
        losses = []
        task_name_list = []
        sum_pi_grads = None # Accumulation of gradients of pi networks

        num_train_epochs = self.args.num_training_epochs
        num_episode_per_epoch = self.dataset.train_num_episode_per_epoch
        num_episode_per_iteration = self.args.num_episodes_per_device * max(1, len(self.device_list)) * self.args.num_iterations_per_optimize_step
        total_iterations = num_episode_per_epoch * num_train_epochs // num_episode_per_iteration
        logger.info(f'Total training iterations: {total_iterations}, num epochs: {num_train_epochs}, num episode per iteration: {num_episode_per_iteration}')

        for iteration in range(self.start_iteration, total_iterations):
            # Each iteration is training on one task

            # Sample a task
            meta_batch = self.dataset.get_train_episode(self.args.num_episodes_per_device, self.device_list)
            task_name_list += meta_batch.task_name_list

            # Train on the task
            outputs = self.model(meta_batch)
            task_loss, task_accuracy = outputs[0], outputs[1]
            if len(task_loss.shape) > 0:
                task_loss = sum(task_loss)
            # Whether to use the output gradients
            use_output_grads = True if len(outputs) > 2 else False
            task_pi_grads = outputs[2] if use_output_grads else None

            task_loss = task_loss / self.args.num_episodes_per_optimize_step
            if not use_output_grads:
                # If gradients are not output, do backward()
                task_loss.backward()
            else:
                # Store the output gradients and apply them in the next optimization step
                if sum_pi_grads is None:
                    sum_pi_grads = task_pi_grads
                else:  # Accumulate all gradients from different episode learner
                    sum_pi_grads = [torch.add(i, j) for i, j in zip(sum_pi_grads, task_pi_grads)]

            # Store results of the current batch
            train_accuracies.append(task_accuracy)
            losses.append(task_loss.item())

            # Optimize & Log
            num_episodes_so_far = (iteration + 1) * self.args.num_episodes_per_device * max(1, len(self.device_list))
            if ( (iteration + 1) % self.args.num_iterations_per_optimize_step == 0) or (iteration == (total_iterations - 1)):
                if not use_output_grads:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    sum_pi_grads = [g.mean(dim=0) for g in sum_pi_grads] 
                    dummy_loss = self.model.dummy_forward(meta_batch)
                    self.optimize_with_pi_gradients(dummy_loss, sum_pi_grads)

                t_loss = sum(losses)
                t_acc = torch.cat(train_accuracies).sum().item() / self.args.num_episodes_per_optimize_step
                
                results = { 'train_loss': t_loss, 'train_acc': t_acc,
                            'iteration': iteration, 'num_episodes': num_episodes_so_far }
                if 'bert-cnap-bn-euc-ar-fim-new' == self.args.model_type:
                    results['l2 loss on gamma/beta'] = self.model.bert.regularization_term().item()

                if not len(task_name_list) == self.args.num_episodes_per_optimize_step == torch.cat(train_accuracies).shape[0]:
                    logger.warning("Number of episode in this optimization step does map the predefined number."
                                   "This may due to break in the middle of an optimization step in preview running.")
                    
                if self.args.wandb is not None:
                    wandb.log(results)
                    for tn, acc in zip(task_name_list, torch.cat(train_accuracies)):
                        wandb.log({f'train_{tn}_acc': acc.item()})
                # logger.debug('\t'.join([f'{k}: {v}' for k, v in results.items()]))
                train_accuracies = []
                losses = []
                task_name_list = []

            # Validate every val_freq optimization steps
            # if (num_episodes_so_far % (self.args.val_freq * self.args.num_episodes_per_optimize_step) == 0) and (iteration + 1) != total_iterations:
            # if ( (iteration + 1) % (self.args.num_iterations_per_optimize_step * self.args.val_freq) == 0) and (iteration != (total_iterations - 1)):
            if (iteration+1)%self.args.val_freq==0 or iteration==(total_iterations-1):
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

                if 'leopard' in self.args.model_type:
                    wandb.log({"leopard inner learning rates":
                                wandb.Histogram(self.model.learning_rates.detach().cpu().numpy())})

            if (iteration + 1) % self.args.checkpoint_freq == 0:
                self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['latest'])
            
            if self.validation_accuracies.early_stop():
                logger.warn(f"Haven't improved for {self.validation_accuracies.earyly_stop_steps} steps. \
                    Stop training and save the current iteration to {self.CHECKPOINT_DIR_NAME['final']}")
                break
        
            if (iteration + 1) % self.PRINT_FREQUENCY == 0:
                logger.info(f'Finished {iteration+1} training iterations.')

        # Save the final model
        self.save_checkpoint(iteration + 1, self.CHECKPOINT_DIR_NAME['final'])

    def clip_grad_norm_(self, grads, max_norm, norm_type=2):
        r"""Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Arguments:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == torch._six.inf:
            total_norm = max(g.data.abs().max() for g in grads)
        else:
            total_norm = 0
            for g in grads:
                param_norm = g.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grads:
                g.data.mul_(clip_coef)
        return total_norm

    def optimize_with_pi_gradients(self, dummy_loss, sum_grads_pi):
        self.optimizer.zero_grad()
        dummy_loss.backward()
        # for g in sum_grads_pi:
        self.clip_grad_norm_(sum_grads_pi, self.args.max_grad_norm)
        with torch.no_grad():
            for p, g in zip(self.model.meta_parameters(), sum_grads_pi):
                assert p.shape == g.shape
                p.grad.copy_(g.data)
        # logger.debug(f'lr: {self.model.learning_rates.grad}')
        self.optimizer.step()
        # logger.debug(f'lr: {self.model.learning_rates}')

    def validate(self):
        # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
        # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
        logger.info("***** Running evaluation *****")
        accuracy_dict ={}
        for batch in self.dataset.val_episode_loop(num_episodes_per_device=self.args.num_episodes_per_device,
                                                       device_list=self.device_list,
                                                       max_num_episode=self.args.max_num_val_episodes):
            loss, acc = self.model(batch, eval=True)
            task_name_list = batch.task_name_list
            assert (acc != -1).sum().item() == len(task_name_list)
            for task_name, acc in zip(task_name_list, acc):
                if task_name in accuracy_dict:
                    accuracy_dict[task_name].append(acc.item())
                else:
                    accuracy_dict[task_name] = [acc.item()]

        for task in accuracy_dict:
            accuracies = accuracy_dict[task]
            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

            accuracy_dict[task] = {"accuracy": accuracy, "confidence": confidence}

        return accuracy_dict

    def run_test(self):
        # test_lasest, test_best: resume previous experiments and test the latest/best model.
        # test_checkpoing: load the model from another experiment given by the checkpoint file path.
        if 'test_latest' in self.args.mode:
            self.resume_from_latest_or_best(load_from_latest=True, exit_if_failed=True)
        elif 'test_best' in self.args.mode:
            self.resume_from_latest_or_best(load_from_latest=False, exit_if_failed=True)
        elif 'test_checkpoint' in self.args.model and self.args.checkpoint_path is not None:
                self.resume_from_checkpoint(self.args.checkpoint_path, model_only=True)
        else:
            raise ValueError('Exp mode not support: ' + self.args.mode)

        self.setup_dataparallel()
        self.test()

    def test(self):
        if '0526' in self.args.mode:
            logger.info('Testing on Version 0526 of Leopard test data')
            test_set = LeopardDataProcessor_0526(self.args, self.tokenizer, self.args.leopard_data_dir_0526)
        else:
            logger.info('Testing on the original Leopard test data')
            test_set = LeopardDataProcessor(self.args, self.tokenizer, self.args.leopard_data_dir)
        self.model.eval()
        device_list = self.device_list if self.parallel_episode_per_device else [self.device_list[0]]

        if '-' in self.args.mode:
            test_shot_list = [int(self.args.mode.split('-')[-1])]
        else:
            test_shot_list = test_set.NUM_SHOTS_LIST

        for num_shots in test_shot_list:
            test_dir = os.path.join(self.args.output_dir, f"leopard-test-{self.start_iteration}-iteration")
            pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True) 

            for task in test_set.TEST_DATASET_LIST:
                r_file = os.path.join(test_dir, f'K{num_shots}_{task}_results.bin')

                # exclude_task_list = None
                # saved_accuracy_dict = None
                if os.path.exists(r_file):
                    logger.info(f'Test results for {num_shots} found in {r_file}, skip testing on this.')
                    continue
                    # saved_accuracy_dict = torch.load(r_file)
                    # exclude_task_list = [k.replace(f'_K{num_shots}', '') for k in saved_accuracy_dict]
                else:
                    specified_task_list = [task]

                previous_res_file = os.path.join(test_dir, f'K{num_shots}_results.bin')
                if os.path.exists(previous_res_file) and f"{task}_K{num_shots}" in torch.load(previous_res_file):
                    logger.info(f"Found {task}_K{num_shots} in previous results, skip testing...")
                    prev_res = torch.load(previous_res_file)
                    accuracy = prev_res[f"{task}_K{num_shots}"]['accuracy']
                    std = prev_res[f"{task}_K{num_shots}"]['std']
                    confidence = prev_res[f"{task}_K{num_shots}"]['confidence']
                else:
                    acc_list = []
                    for meta_batch in test_set.episode_loop(self.args.num_episodes_per_device,
                                                            device_list, num_shots, validation=False,
                                                            specified_task_list=specified_task_list):
                        _, acc = self.model(meta_batch, eval=True)
                        task_name_list = meta_batch.task_name_list
                        assert (acc != -1).sum().item() == len(task_name_list)
                        for task_name, acc in zip(task_name_list, acc):
                            assert acc != -1
                            acc_list.append(acc.item())
                            logger.info(f'Test task: {task_name}, {num_shots} shots, acc: {acc.item()}')

                    accuracy = np.array(acc_list).mean() * 100.0
                    std = np.array(acc_list).std() * 100.0
                    confidence = (196.0 * std) / np.sqrt(len(acc_list))

                r_dict = {"accuracy": accuracy, "confidence": confidence, "std": std}
                logger.info(f'Test task: {task}, mean acc: {accuracy}, std: {std}, confidence: {confidence}')

                logger.info(f'Saving testing results to {r_file}')
                torch.save(r_dict, r_file)
                # with open(os.path.join(test_dir, f'{num_shots}_shot_slurm_id'), 'w+') as f:
                #       f.write(self.args.slurm_job_id)

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
    parser.add_argument("--leopard_data_dir_0526", default=None, type=str, required=False,
                        help="The dir of Leopard test datasets that are processed on 2020/05/26. Long sequences are removed.")

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
    parser.add_argument("--cnap_freeze_linear_layer", default=True, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to freeze the last linear layer in second stage training.")
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

    parser.add_argument("--use_improve_loss", default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to use the loss before and after adaptation as the final loss.")

    return parser

def main():
    # Arg parser initialization and parsing
    parser = init_arg_parser(MetaLearner)
    args = parser.parse_args()
    if args.wandb.strip().lower() == 'none':
        args.wandb = None

    learner = MetaLearner(args)
    learner.run()

if __name__ == "__main__":
    main()
