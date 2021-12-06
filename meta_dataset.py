"""Dataloader for meta-training datasets."""

import itertools
import os
from transformers.data.processors.utils import InputFeatures
from utils import get_device_name
import torch
import numpy as np
from torch.utils.data import (DataLoader, TensorDataset, dataloader)
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
from transformers import glue_output_modes
from transformers import glue_processors
from transformers import glue_convert_examples_to_features

from logging_utils import get_logger
logger = get_logger('Meta-Data-Loader')

TASK_TEXT_LABELS = {}
TASK_TEXT_LABELS['MNLI'] = ["contradiction", "entailment", "neutral"]
TASK_TEXT_LABELS['MRPC'] = ["not paraphase", "paraphase"]
TASK_TEXT_LABELS['SST-2'] = ["negative movie review", "positive movie review"]
TASK_TEXT_LABELS['QQP'] = ["not paraphase", "paraphase"]
TASK_TEXT_LABELS['QNLI'] = ["entailment", "not entailment"] 
TASK_TEXT_LABELS['RTE'] = ["entailment", "not entailment"] 
TASK_TEXT_LABELS['SNLI'] = ["contradiction", "entailment", "neutral"] 
TASK_TEXT_LABELS['CoLA'] = ['not grammatically acceptable', 'grammatically acceptable']
TASK_TEXT_LABELS['SciTail'] = ["entailment", "neutral"] 

def load_and_cache_examples(args, data_dir, task, tokenizer, split):
    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        split,
        args.lm_type,
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if split == 'val':
            examples = processor.get_dev_examples(data_dir) 
        elif split == 'train':
            examples = processor.get_train_examples(data_dir)
        elif split == 'test':
            pass
        else:
            raise ValueError(f'Unsupported split: {split}')

        features = glue_convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=False,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    logger.info('Convert to Tensors and build dataset')
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    logger.info(f'Finish loading {task}')
    return dataset
        
class ClassBalancedRandomSampler(Sampler):
    """Samples elements randomly with class balance.

    Attributes:
        data_source_labels (list): List of dataset labels.
        strict_balance (bool): If true, every batch with the size of N * num_class
          will be class balanced. For example:
        disturb: Whether to disturb the dataset.

            Class 1: * * * * * * * *
            Class 2: # # # # # # # # #
        
          if the batch size is 3 * 2 = 6, only the last batch is discarded since 
          it's not class balanced:
        
            Class 1: |* * *|* * *|* *
            Class 2: |# # #|# # #|# # #
                                    |-> discarded
    """

    def __init__(self, data_source_labels, strict_balance=False, disturb=True):
        self.data_source_labels = data_source_labels
        self.strict_balance = strict_balance
        self.distrub = disturb
            
    def __iter__(self):
        """ 
            Here we permutate the index of each class separately and then merge 
            the indexes so that the correspoding label sequence looks like:
                    # * # * # * # * # * # * # * # * # * ...
            Sample batches sequencially with size of N * num_class with result in
            class balanced batches, except the last few batches depending on how
            balanced the dataset is.
        """
        unique_labels = list(set(self.data_source_labels))
        label_list = np.array(self.data_source_labels)

        perm_list = []
        label_idx = {}
        for l in unique_labels:
            idx = np.where(label_list == l)[0]
            if self.distrub:
                idx = np.random.permutation(idx)
            label_idx[l] = idx.tolist()
        # use min to make sure every class is include in each batch with size of N * num_class
        min_or_max = min if self.strict_balance else max
        size = min_or_max([len(label_idx[l]) for l in label_idx])
        for _ in range(size):
            for l in label_idx:
                if len(label_idx[l]) > 0:
                    perm_list.append(label_idx[l].pop())
        return iter(perm_list)

    def __len__(self):
        return len(self.data_source_labels)

class MetaBatch:
    """Used for parallel episodic training with DataParallel.
    
    One episode on each device. Note we can't simply feed a dictionary into
    DataParallel because each value will be split along the first dimension. The
    attributes of an object will not be split.
    """
    def __init__(self, device_batch_dict, task_name_list=None):
        for device_name in device_batch_dict:
            setattr(self, device_name, device_batch_dict[device_name])
        self.task_name_list = task_name_list

class DatasetProcessor:
    """Abstract class for dataset processor.
    Attributes:
        args
        tokenizer: e.g. BERT tokenizer
        train_task_list (list): List of tasks for meta-training.
        val_task_list (list): List of tasks for meta-validation.
        test_task_list (list): List of tasks for meta-testing.
    """
    def __init__(self, args, tokenizer, train_task_list, val_task_list, test_task_list):
        self.args = args
        self.tokenizer = tokenizer
        self.split_task_list = {
            'train': train_task_list,
            'val': val_task_list,
            # 'test': test_task_list
        }
        
        self.init_dataloader()

    def init_dataloader(self):
        raise NotImplementedError()

    def features_to_tensors(self, features):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        return (all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    def text_to_features(self, text, label):
        mask_padding_with_zero = True
        max_length=self.args.max_seq_length
        pad_on_left = False
        pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        pad_token_segment_id=0

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        return InputFeatures(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label=label)

    def _episode_generator(self, dataloader, infinite_loop=True):
        if infinite_loop:
            while True:
                for episode in dataloader:
                    yield episode
        else:
            for episode in dataloader:
                yield episode

    def _task_generator(self, task_list, sample_weights=None):
        if sample_weights is None: 
            while True:
                yield np.random.choice(task_list)
        else:
            if len(sample_weights) != len(task_list):
                raise ValueError('Count of sampling weights and tasks must match.')
            if abs(sum(sample_weights) - 1) > 0.0001:
                raise ValueError('Sampling weights need to be normalized.')
            
            while True:
                for i in WeightedRandomSampler(sample_weights, 100, replacement=True):
                    yield task_list[i]

    def _prepare_episode(self, batch, task_id=None, label_features=None, text_labels=None, device=None):
        """ Batch -> Episode

        Args:
            batch (tuple<torch.Tensor>): First half is the support set; second
            half is the query set.

        Returns:
            dict: Data for this episode.
        """
        if task_id is not None:
            task_id = torch.tensor(task_id, dtype=torch.int)
        if device is not None:
            batch = tuple(t.to(device) for t in batch)

        # num_examples = batch[0].shape[0]
        # total_num_batches = num_query_batches + num_support_batches
        # num_support = num_examples * num_support_batches // total_num_batches
        if batch[3].max() + 1 != batch[3].unique().shape[0]:
            raise ValueError('Largest class id should match number of classes.')

        episode =  {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3] }

        if task_id is not None:
            episode['task_id'] = task_id.to(device) if device is not None else task_id
        if label_features is not None:
            label_features = tuple(t.to(device) for t in label_features) if device is not None else label_features
            episode['label_features'] = label_features
        if text_labels is not None:
            episode['text_labels'] = text_labels

        return episode

class MetaDatasetProcessor(DatasetProcessor):
    """ Dataset processor for meta-training on GLUE tasks.  """
    def __init__(self, args, tokenizer, train_task_list, val_task_list, test_task_list):
        super().__init__(args, tokenizer, train_task_list, val_task_list, test_task_list)

    def _init_dataloader_of_split(self, task_name, split):
        logger.info(f'***** Loading data for {split} *****')
        if task_name.lower() not in glue_processors:
            raise ValueError("Task not found: %s" % (task_name))
        processor = glue_processors[task_name.lower()]()
        # output_mode = glue_output_modes[task_name.lower()]
        batch_size = (self.args.num_shots_support+self.args.num_shots_query) * len(processor.get_labels())

        data_dir = os.path.join(self.args.data_dir, task_name)
        dataset = load_and_cache_examples(self.args, data_dir, task_name.lower(),
                                        self.tokenizer, split)

        dataloader = DataLoader(dataset,
                                sampler=ClassBalancedRandomSampler(dataset.tensors[-1].tolist(), strict_balance=True),
                                batch_size=batch_size,
                                drop_last=True) 
        return dataloader, len(dataset)

    def init_dataloader(self):
        self.train_episode_gen_dict = {}
        self._train_task_id_dict = {}
        for i, task in enumerate(self.split_task_list['train']):
            self._train_task_id_dict[task] = i
        self._val_task_id_dict = {}
        for i, task in enumerate(self.split_task_list['val']):
            self._val_task_id_dict[task] = i

        val_dataloader_dict = {}
        test_dataloader_dict = {}
        task_train_size = []
        total_num_batches_each_episode = self.args.num_query_batches + self.args.num_support_batches

        train_total_episode = 0
        val_total_episode = 0
        test_total_episode = 0
        
        # label feature generation
        logger.info("Generating task label features")
        self.task_label_features = {}
        all_tasks = set(itertools.chain.from_iterable(self.split_task_list.values()))
        for task_name in set(all_tasks):
            self.task_label_features[task_name] = self.features_to_tensors([
                self.text_to_features(' # '.join(TASK_TEXT_LABELS[task_name]), -1)
            ])
        
        self.split_dataloader = {}
        train_task_size_list = []
        for split, task_list in self.split_task_list.items():
            logger.info(f'***** Loading split: {split} *****')
            dataloader_dict = {}
            total_episode = 0
            for task_name in task_list:
                dataloader_dict[task_name], task_size = self._init_dataloader_of_split(task_name, split)
                total_episode += len(dataloader_dict[task_name])

                if split == 'train':
                    self.train_episode_gen_dict[task_name] = self._episode_generator(dataloader_dict[task_name])
                    train_task_size_list.append(task_size)

            setattr(self, f'{split}_num_episode_per_epoch', total_episode)                
            self.split_dataloader[split] = dataloader_dict

            if split == 'train':
                # Sample dataset according sqrt of data size.
                train_task_size_list = np.array(train_task_size_list)
                train_task_size_list = np.sqrt(train_task_size_list)
                train_task_size_list = train_task_size_list / np.sum(train_task_size_list)
                self._train_task_gen = self._task_generator(task_list,
                                                           sample_weights=train_task_size_list.tolist())
 
    def get_train_episode(self, num_episodes_per_device, device_list): 
        """ Get data of one episode. """
        task_index = 0
        device_batch_dict = {}
        task_name_list = []
        for device in device_list:
            episode_list = []
            for _ in range(num_episodes_per_device):
                task_name = next(self._train_task_gen)
                batch = next(self.train_episode_gen_dict[task_name])
                episode = self._prepare_episode(batch,
                                                task_id=self._train_task_id_dict[task_name],
                                                label_features=self.task_label_features[task_name],
                                                text_labels=TASK_TEXT_LABELS[task_name],
                                                device=device)
                task_index += 1
                episode_list.append(episode)
                task_name_list.append(task_name)
            device_batch_dict[get_device_name(device)] = episode_list
            # device_batch_dict[get_device_name(device)] = {k: torch.cat([episode[k] for episode in episode_list], dim=0)
            #                                             for k in episode_list[0]}     
        return MetaBatch(device_batch_dict, task_name_list=task_name_list)

    def get_train_episode_different_task_on_each_device(self, num_episodes_per_device, device_list): 
        """ Get data of one episode. """
        task_index = 0
        device_batch_dict = {}
        for device in device_list:
            episode_list = []
            saved_task = []
            for _ in range(num_episodes_per_device):
                while True:
                    task_name = next(self._train_task_gen)
                    if task_name not in saved_task:
                        saved_task.append(task_name)
                        break
                batch = next(self._train_episode_gen_dict[task_name])
                episode = self._prepare_episode(batch,
                                                task_id=self._train_task_id_dict[task_name],
                                                label_features=self.task_label_features[task_name],
                                                text_labels=TASK_TEXT_LABELS[task_name],
                                                device=device)
                task_index += 1
                episode_list.append(episode)
            device_batch_dict[get_device_name(device)] = episode_list
            # device_batch_dict[get_device_name(device)] = {k: torch.cat([episode[k] for episode in episode_list], dim=0)
            #                                             for k in episode_list[0]}     
        return MetaBatch(device_batch_dict)

    def val_episode_loop(self, num_episodes_per_device, device_list, max_num_episode=-1):
        device_batch_dict = {}
        task_index = 0
        task_name_list = []

        for task_name in self.split_task_list['val']:
            count = 0 
            for batch in self.split_dataloader['val'][task_name]:
                device_idx = task_index // num_episodes_per_device
                device_name = get_device_name(device_list[device_idx])
                episode = self._prepare_episode(batch,
                                                task_id=self._val_task_id_dict[task_name],
                                                label_features=self.task_label_features[task_name],
                                                text_labels=TASK_TEXT_LABELS[task_name],
                                                device=device_list[device_idx])
                if not device_name in device_batch_dict:
                    device_batch_dict[device_name] = [episode]
                else:
                    device_batch_dict[device_name].append(episode)
                task_name_list.append(task_name)
                task_index += 1
                if task_index == num_episodes_per_device * len(device_list):
                    yield MetaBatch(device_batch_dict, task_name_list=task_name_list)
                    device_batch_dict.clear()
                    task_index = 0
                    task_name_list = []

                count += 1
                if max_num_episode > 0 and count == max_num_episode:
                    break
        if task_index > 1:
            yield MetaBatch(device_batch_dict, task_name_list=task_name_list)

    def val_episode_loop_different_task_on_each_device(self, num_episodes_per_device, device_list):
        epi_count = 0
        max_epi_count = 50

        task_batch_iter_dict = {task:iter(loader) for task,loader in self._val_dataloader_dict.items()}
        while True:
            device_batch_dict = {}
            task_name_list = []
            for device_idx in range(len(device_list)):
                device_task_list = []
                device_task_batch_dict = {}
                for task in self._val_list:
                    if len(device_task_list) < num_episodes_per_device:
                        batch = next(task_batch_iter_dict[task], None)
                        if batch is not None:
                            device_task_list.append(task)
                            device_task_batch_dict[task] = batch
                if len(device_task_list) == num_episodes_per_device:
                    for task_name in device_task_list:
                        device_name = get_device_name(device_list[device_idx])
                        batch = device_task_batch_dict[task_name]
                        episode = self._prepare_episode(batch,
                                                        task_id=self._val_task_id_dict[task_name],
                                                        label_features=self.task_label_features[task_name],
                                                        text_labels=TASK_TEXT_LABELS[task_name],
                                                        device=device_list[device_idx])
                        if not device_name in device_batch_dict:
                            device_batch_dict[device_name] = [episode]
                        else:
                            device_batch_dict[device_name].append(episode)
                        task_name_list.append(task_name)
            if len(device_batch_dict) > 0:
                epi_count += 1
                yield MetaBatch(device_batch_dict, task_name_list=task_name_list)
                if epi_count == max_epi_count:
                    break
            else:
                break
         
class RegularDatasetProcessor(DatasetProcessor):
    def __init__(self, args, tokenizer, train_task_list, val_task_list, test_task_list):
        super().__init__(args, tokenizer, train_task_list, val_task_list, test_task_list)
    
    def _init_dataloader_of_split(self, task_name, split):
        logger.info(f'***** Loading data for {split} *****')
        if task_name.lower() not in glue_processors:
            raise ValueError("Task not found: %s" % (task_name))
        processor = glue_processors[task_name.lower()]()
        # output_mode = glue_output_modes[task_name.lower()]
        batch_size = (self.args.num_shots_support+self.args.num_shots_query) * len(processor.get_labels())

        data_dir = os.path.join(self.args.data_dir, task_name)
        dataset = load_and_cache_examples(self.args, data_dir, task_name.lower(),
                                        self.tokenizer, split)

        dataloader = DataLoader(dataset,
                                sampler=ClassBalancedRandomSampler(dataset.tensors[-1].tolist(), strict_balance=True),
                                batch_size=batch_size,
                                drop_last=True) 
        return dataloader, len(dataset)
    
    def init_dataloader(self):
        # label feature generation
        logger.info("Generating task label features")
        self.task_label_features = {}
        all_tasks = set(itertools.chain.from_iterable(self.split_task_list.values()))
        for task_name in all_tasks:
            self.task_label_features[task_name] = self.features_to_tensors([
                self.text_to_features(' # '.join(TASK_TEXT_LABELS[task_name]), -1)
            ])

        self.split_dataloader = {}
        self.train_task_gen = None
        self.train_episode_gen_dict = {}
        train_task_size_list = []
        for split, task_list in self.split_task_list.items():
            logger.info(f'Loading split: {split}')
            dataloader_dict = {}
            total_batches = 0

            for task_name in task_list:
                dataloader, task_size = self._init_dataloader_of_split(task_name, split)
                dataloader_dict[task_name] = dataloader
                total_batches += len(dataloader)
                logger.info(f'Loaded dataset {task_name}. Batches # per epoch is {len(dataloader)}')

                if split == 'train':
                    self.train_episode_gen_dict[task_name] = self._episode_generator(dataloader)
                    train_task_size_list.append(task_size)

            self.split_dataloader[split] = dataloader_dict

            if split == 'train':
                # Sample dataset according sqrt of data size
                train_task_size_list = np.array(train_task_size_list)
                train_task_size_list = np.sqrt(train_task_size_list)
                train_task_size_list = train_task_size_list / np.sum(train_task_size_list)
                self.train_task_gen = self._task_generator(task_list,
                                                           sample_weights=train_task_size_list.tolist())

    def get_episodes_from_different_tasks_on_each_device(self, num_episodes_per_device, device_list): 
        """ Get data of one episode. """
        task_index = 0
        device_batch_dict = {}
        task_name_list = []
        for device in device_list:
            episode_list = []
            saved_task = set()
            for _ in range(num_episodes_per_device):
                while True:
                    task_name = next(self.train_task_gen)
                    if task_name not in saved_task:
                        saved_task.add(task_name)
                        task_name_list.append(task_name)
                        break
                batch = next(self.train_episode_gen_dict[task_name])
                episode = self._prepare_episode(batch,
                                                label_features=self.task_label_features[task_name],
                                                text_labels=TASK_TEXT_LABELS[task_name],
                                                device=device)
                task_index += 1
                episode_list.append(episode)
            device_batch_dict[get_device_name(device)] = episode_list
        return MetaBatch(device_batch_dict, task_name_list)

class TaskDataset(DatasetProcessor):
    """ Dataset of tasks.
        
        Each datapoint is a task. Each task is a mini-dataset, consisting of N 
        classes and a few examples per class.
    """
    def __init__(self, args, tokenizer, train_task_list, val_task_list, test_task_list):
        super().__init__(args, tokenizer, train_task_list, val_task_list, test_task_list)

    def _init_dataloader_of_split(self, task_name, split):
        logger.info(f'***** Loading data for {split} *****')
        if task_name.lower() not in glue_processors:
            raise ValueError("Task not found: %s" % (task_name))
        processor = glue_processors[task_name.lower()]()
        # output_mode = glue_output_modes[task_name.lower()]

        # Each batch is actually a pair of tasks with the same size.
        batch_size = len(processor.get_labels())

        data_dir = os.path.join(self.args.data_dir, task_name)
        dataset = load_and_cache_examples(self.args, data_dir, task_name.lower(),
                                        self.tokenizer, split)

        dataloader = DataLoader(dataset,
                                sampler=ClassBalancedRandomSampler(dataset.tensors[-1].tolist(), strict_balance=True),
                                batch_size=batch_size,
                                drop_last=True) 
        return dataloader, len(dataset)

    def init_dataloader(self):
        self.train_episode_gen_dict = {}
        self._train_task_id_dict = {}
        for i, task in enumerate(self.split_task_list['train']):
            self._train_task_id_dict[task] = i
        self._val_task_id_dict = {}
        for i, task in enumerate(self.split_task_list['val']):
            self._val_task_id_dict[task] = i

        # label feature generation
        logger.info("Generating task label features")
        self.task_label_features = {}
        all_tasks = set(itertools.chain.from_iterable(self.split_task_list.values()))
        for task_name in set(all_tasks):
            self.task_label_features[task_name] = self.features_to_tensors([
                self.text_to_features(' # '.join(TASK_TEXT_LABELS[task_name]), -1)
            ])
        
        self.split_dataloader = {}
        train_task_size_list = []
        for split, task_list in self.split_task_list.items():
            logger.info(f'***** Loading split: {split} *****')
            dataloader_dict = {}
            total_episode = 0
            for task_name in task_list:
                dataloader_dict[task_name], task_size = self._init_dataloader_of_split(task_name, split)
                total_episode += len(dataloader_dict[task_name])

                if split == 'train':
                    self.train_episode_gen_dict[task_name] = self._episode_generator(dataloader_dict[task_name])
                    train_task_size_list.append(task_size)

            setattr(self, f'{split}_num_episode_per_epoch', total_episode)                
            self.split_dataloader[split] = dataloader_dict

            if split == 'train':
                # Sample dataset according sqrt of data size.
                train_task_size_list = np.array(train_task_size_list)
                train_task_size_list = np.sqrt(train_task_size_list)
                train_task_size_list = train_task_size_list / np.sum(train_task_size_list)
                self._train_task_gen = self._task_generator(task_list,
                                                           sample_weights=train_task_size_list.tolist())
 
    def get_train_episode_different_task_on_each_device(self, num_episodes_per_device, device_list, min_shots, max_shots, num_per_task=2): 
        """ Get data of one episode. """
        task_index = 0
        device_batch_dict = {}
        for device in device_list:
            episode_list = []
            saved_task = [] # Make sure the tasks on one device are distinct with each other. 
            for _ in range(num_episodes_per_device):
                # Make sure not sampling the same task.
                while True:
                    task_name = next(self._train_task_gen)
                    if task_name not in saved_task:
                        saved_task.append(task_name)
                        break

                episode_cur_task = []
                for _ in range(num_per_task):
                    num_shots = np.random.randint(min_shots, max_shots+1)
                    batch_list = []
                    for _ in range(num_shots):
                        # Each batch is a one shot task.
                        batch_list.append(next(self.train_episode_gen_dict[task_name]))
                    batch = self._merge_batches(batch_list)

                    episode = self._prepare_episode(batch,
                                                    task_id=self._train_task_id_dict[task_name],
                                                    label_features=self.task_label_features[task_name],
                                                    text_labels=TASK_TEXT_LABELS[task_name],
                                                    device=device)
                    episode_cur_task.append(episode)
                    task_index += 1
                episode_list.append(episode_cur_task)
            device_batch_dict[get_device_name(device)] = episode_list

        return MetaBatch(device_batch_dict)

    def _merge_batches(self, batches):
        return [torch.cat([b[i] for b in batches], dim=0) for i in range(len(batches[0]))]