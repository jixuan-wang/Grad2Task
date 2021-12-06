# coding=utf-8
""" Multi-task meta training. """

from __future__ import absolute_import, division, print_function

import argparse
import os
import pathlib
import numpy as np
from itertools import chain
from shutil import Error

import torch
from sklearn.metrics import roc_auc_score

from leopard_data import LeopardDataProcessor
from modeling_baselines import FineTuningBERT, FineTuningBERTWithBN, FineTuningBERTWithFilm, FineTuningProtoNet, FineTuningProtoNetBN, FineTuningProtoNetFilm

import wandb

from logging_utils import get_logger
from main_meta_training import ALL_MODELS
from main_meta_training import MetaLearner
from main_meta_training import init_arg_parser

logger = get_logger('Baseline')

class Learner(MetaLearner):
    """ Main class for training, evaluation and testing. """
    MODEL_CLASSES = {
        'fine-tune-bert': FineTuningBERT,
        'fine-tune-bert-bn': FineTuningBERTWithBN,
        'fine-tune-bert-film': FineTuningBERTWithFilm,
        'fine-tune-protonet': FineTuningProtoNet,
        'fine-tune-protonet-bn': FineTuningProtoNetBN,
        'fine-tune-protonet-bn-film': FineTuningProtoNetFilm
    }

    def run_test(self):
        self.test()

    def hyper_param_tuning(self):
        file_for_save = os.path.join(self.args.output_dir, "best_finetune_hyper_param.pt")
        cached_dir = os.path.join(self.args.output_dir, f"hyper_param_tuning_cached")
        pathlib.Path(cached_dir).mkdir(parents=True, exist_ok=True) 
        if not os.path.exists(file_for_save):
            dataset = self._load_train_data()
            best_acc = 0.
            best_lr = 0
            best_num_epoch = 0
            for learning_rate in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
                for num_epoch in [5, 10, 15, 20]:
                    cached_file = os.path.join(cached_dir, f'{learning_rate}-{num_epoch}.pt')
                    if not os.path.exists(cached_file):
                        self.args.fine_tune_epochs = num_epoch
                        self.args.learning_rate = learning_rate

                        test_loss_list = []
                        test_acc_list = []

                        for batch in dataset.val_episode_loop(num_episodes_per_device=self.args.num_episodes_per_device,
                                                            device_list=self.device_list,
                                                            max_num_episode=10):
                            test_loss, test_acc = self.model(batch, eval=True, test_per_epoch=False)
                            task_name_list = batch.task_name_list

                            valid_num = 0
                            for acc in test_acc:
                                if acc.sum() != -1:
                                    valid_num += 1
                                else:
                                    break
                            assert valid_num == len(task_name_list)
                            r_num = len(task_name_list)
                            test_loss_list.append(test_loss[:r_num])
                            test_acc_list.append(test_acc[:r_num])
                            logger.info(f"Task {task_name_list}, acc {test_acc}")
                        
                        avg_acc = torch.cat(test_acc_list).mean().item()
                        torch.save(avg_acc, cached_file)
                    else:
                        avg_acc = torch.load(cached_file)

                    if avg_acc > best_acc:
                        best_lr = learning_rate
                        best_num_epoch = num_epoch
                        best_acc = avg_acc
            
            self.args.fine_tune_epochs = best_num_epoch
            self.args.learning_rate = best_lr
            torch.save({
                "lr": best_lr,
                "num_epochs": best_num_epoch
            }, file_for_save)
        else:
            saved = torch.load(file_for_save)
            best_lr = saved['lr']
            best_num_epoch = saved['num_epochs']

        logger.info(f"lr: {best_lr}, num_epochs: {best_num_epoch}")
        if self.args.wandb is not None:
            wandb.config.best_lr = best_lr
            wandb.config.best_num_epochs = best_num_epoch
            wandb.log({"test": 100})

    def test(self):
        if self.args.wandb is not None:
            wandb.init(project=self.args.wandb)
            wandb.config.model_type = self.args.model_type
            wandb.config.slurm_job_id = self.args.slurm_job_id
        
        self.hyper_param_tuning()

        test_set = LeopardDataProcessor(self.args, self.tokenizer, self.args.leopard_data_dir)
        device_list = self.device_list if self.parallel_episode_per_device else [self.device_list[0]]
        test_per_epoch = self.args.test_per_epoch.lower() == 'true'

        if '-' in self.args.mode:
            test_shot_list = [int(self.args.mode.split('-')[-1])]
        else:
            test_shot_list = test_set.NUM_SHOTS_LIST

        for num_shots in test_shot_list:
            test_dir = os.path.join(self.args.output_dir, f"test-{self.start_iteration}-iteration")
            pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True) 

            for task in test_set.TEST_DATASET_LIST:
                r_file = os.path.join(test_dir, f'K{num_shots}_{task}_results.bin')

                # exclude_task_list = None
                # saved_accuracy_dict = None
                if os.path.exists(r_file):
                    logger.info(f'Test results for {num_shots} found in {r_file}, skip testing on this.')
                    res = torch.load(r_file)
                    acc_list = res['test_acc']
                    if self.args.wandb is not None:
                        accuracy = acc_list.mean() * 100.0
                        std = acc_list.std() * 100.0
                        confidence = (196.0 * std) / np.sqrt(len(acc_list))
                        wandb.log({
                            f"K{num_shots}_{task}_acc": accuracy,
                            f"K{num_shots}_{task}_std": std,
                            f"K{num_shots}_{task}_conf": confidence 
                        })
                    continue
                    # saved_accuracy_dict = torch.load(r_file)
                    # exclude_task_list = [k.replace(f'_K{num_shots}', '') for k in saved_accuracy_dict]
                else:
                    specified_task_list = [task]

                acc_list = []
                task_idx = 0
                train_loss_list = []
                train_acc_list = []
                test_loss_list = []
                test_acc_list = []
                all_task_list = []
                
                for meta_batch in test_set.episode_loop(self.args.num_episodes_per_device,
                                                        device_list, num_shots, validation=False,
                                                        specified_task_list=specified_task_list):
                    if test_per_epoch:
                        train_loss, train_acc, test_loss, test_acc = self.model(meta_batch, eval=True, test_per_epoch=True)
                        task_name_list = meta_batch.task_name_list

                        valid_num = 0
                        for acc in test_acc:
                            if acc.sum() != -1:
                                valid_num += 1
                            else:
                                break
                        assert valid_num == len(task_name_list)
                        r_num = len(task_name_list)
                        train_loss_list.append(train_loss[:r_num])
                        train_acc_list.append(train_acc[:r_num])
                        test_loss_list.append(test_loss[:r_num])
                        test_acc_list.append(test_acc[:r_num])
                    else:
                        test_loss, test_acc = self.model(meta_batch, eval=True, test_per_epoch=False)
                        task_name_list = meta_batch.task_name_list

                        valid_num = 0
                        for acc in test_acc:
                            if acc.sum() != -1:
                                valid_num += 1
                            else:
                                break
                        assert valid_num == len(task_name_list)
                        r_num = len(task_name_list)
                        test_loss_list.append(test_loss[:r_num])
                        test_acc_list.append(test_acc[:r_num])
                        all_task_list += task_name_list
                        # logger.info(f"Task {task_name_list}, acc {test_acc}")
                            
                # accuracy = np.array(acc_list).mean() * 100.0
                # std = np.array(acc_list).std() * 100.0
                # confidence = (196.0 * std) / np.sqrt(len(acc_list))

                # r_dict = {"accuracy": accuracy, "confidence": confidence, "std": std}
                # logger.info(f'Test task: {task}, mean acc: {accuracy}, std: {std}, confidence: {confidence}')

                logger.info(f'Saving testing results to {r_file}')
                if test_per_epoch:
                    torch.save({
                        'train_loss': torch.stack(train_loss_list).cpu().numpy(),
                        'train_acc': torch.stack(train_acc_list).cpu().numpy(),
                        'test_loss': torch.stack(test_loss_list).cpu().numpy(),
                        'test_acc': torch.stack(test_acc_list).cpu().numpy()
                        }, r_file)
                else:
                    torch.save({
                        'test_loss': torch.cat(test_loss_list).cpu().numpy(),
                        'test_acc': torch.cat(test_acc_list).cpu().numpy()
                        }, r_file)
                    if self.args.wandb is not None:
                        acc_list = torch.cat(test_acc_list).cpu().numpy()
                        accuracy = acc_list.mean() * 100.0
                        std = acc_list.std() * 100.0
                        confidence = (196.0 * std) / np.sqrt(len(acc_list))
                        wandb.log({
                            f"K{num_shots}_{task}_acc": accuracy,
                            f"K{num_shots}_{task}_std": std,
                            f"K{num_shots}_{task}_conf": confidence 
                        })
                    
                # with open(os.path.join(test_dir, f'{num_shots}_shot_slurm_id'), 'w+') as f:
                #       f.write(self.args.slurm_job_id)

def main():
    # Arg parser initialization and parsing
    parser = init_arg_parser(Learner)
    parser.add_argument('--bert_pretrained', default=None, type=str,
                        help="Pretrained BERT with bottleneck adapters.")
    parser.add_argument('--test_per_epoch', default='false', type=str,
                        help="Set this flag to test after each fine-tuning epoch.")
    args = parser.parse_args()
    if args.wandb.strip().lower() == 'none':
        args.wandb = None

    learner = Learner(args)
    learner.run()

if __name__ == "__main__":
    main()
