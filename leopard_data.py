from meta_dataset import MetaBatch
import os
import json
import numpy as np
from utils import get_device_name
import torch
from torch.utils.data.dataset import TensorDataset
from transformers.data import DataProcessor, InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features

from logging_utils import get_logger
logger = get_logger('Leopard-Eval-DataLoader')


class LeopardDataProcessor(DataProcessor):
    TEST_DATASET_LIST = ['airline', 'emotion', 'political_message',
                        'rating_electronics', 'conll_c', 'political_audience',
                        'rating_books', 'rating_kitchen', 'sentiment_books',
                        'sentiment_kitchen', 'disaster', 'political_bias',
                        'rating_dvd', 'restaurant', 'sentiment_dvd', 'scitail']
    TEST_DATASET_LIST += ['huffpost_10', 'yelp', 'snips'] 
    VAL_DATASET_LIST = ['scitail', 'sentiment_electronics']

    TRAIN_SET_INDEX_LIST = list(range(10))
    NUM_SHOTS_LIST = [4, 8, 16]

    def __init__(self, args, tokenizer, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.args = args
        self.tokenizer = tokenizer
        self.test_dataset = self.load_and_cache_examples()
        self.val_dataset = self.load_and_cache_examples(validation=True)

    def _task_generator(self, num_shots, validation=False, exclude_task_list=None, specified_task_list=None):
        if num_shots not in self.NUM_SHOTS_LIST:
            raise ValueError(f'Only numbers of shots in {self.NUM_SHOTS_LIST} are supported')

        dataset = self.val_dataset if validation else self.test_dataset
        task_list = self.VAL_DATASET_LIST if validation else self.TEST_DATASET_LIST

        if exclude_task_list is not None and specified_task_list is not None:
            raise ValueError('exclude_task_list and specified_task_list can not be non-empty at the same time.')
        if not exclude_task_list is None:
            task_list = [ t for t in task_list if t not in exclude_task_list]
        if not specified_task_list is None:
            for t in specified_task_list:
                if not t in task_list:
                    raise ValueError(f'Unknown task: {t}')
            task_list = specified_task_list
        
        for task in task_list: 
            query_set = dataset[task]['test']
            for i in self.TRAIN_SET_INDEX_LIST:
                support_set = dataset[task]['train'][f'{i}_{num_shots}']
                if support_set[0].shape[0] == 0:
                    logger.warn(f'Empty support set found for {task} {i} {num_shots}. Skip it.')
                    continue
                yield (task, support_set, query_set, dataset[task]['label_features'], dataset[task]['label_list'])

    def _task_generator_by_task(self, task, num_shots, validation=False):
        if num_shots not in self.NUM_SHOTS_LIST:
            raise ValueError(f'Only numbers of shots in {self.NUM_SHOTS_LIST} are supported')

        dataset = self.val_dataset if validation else self.test_dataset
        
        query_set = dataset[task]['test']
        for i in self.TRAIN_SET_INDEX_LIST:
            support_set = dataset[task]['train'][f'{i}_{num_shots}']
            if support_set[0].shape[0] == 0:
                logger.warn(f'Empty support set found for {task} {i} {num_shots}. Skip it.')
                continue
            yield (task, support_set, query_set, dataset[task]['label_features'], dataset[task]['label_list'])

    def episode_loop(self, num_episodes_per_device, device_list, num_shots, validation, exclude_task_list=None, specified_task_list=None):
        device_batch_dict = {}
        task_index = 0
        task_name_list = []

        for (task_name, support_set, query_set, label_features, labels) in self._task_generator(num_shots, validation, exclude_task_list, specified_task_list):
            device_idx = task_index // num_episodes_per_device
            device_name = get_device_name(device_list[device_idx])
            episode = self._prepare_episode(support_set, query_set, num_shots,
                                            label_features,
                                            labels,
                                            device_list[device_idx])
            if device_name in device_batch_dict:
                device_batch_dict[device_name].append(episode)
            else:
                device_batch_dict[device_name] = [episode]
            task_name_list.append(task_name)
            task_index += 1
            if task_index == num_episodes_per_device * len(device_list):
                yield MetaBatch(device_batch_dict, task_name_list=task_name_list)
                device_batch_dict.clear()
                task_index = 0
                task_name_list = []

        if task_index > 1:
            yield MetaBatch(device_batch_dict, task_name_list=task_name_list)

    def episode_loop_for_taskemb(self, num_shots, num_episodes_per_device, device_list):
        assert len(device_list) == 1
        device = device_list[0]

        task_name_list = []

        dataset = self.test_dataset
        task_support_list = {}
        for i in range(len(self.TEST_DATASET_LIST)):
            l = list(range(len(self.TRAIN_SET_INDEX_LIST)))
            np.random.shuffle(l)
            task_support_list[i] = l

        task_ind_list = list(range(len(self.TEST_DATASET_LIST)))
        task_ind_list = [t for t in task_support_list if len(task_support_list[t]) >= 2]
        while len(task_ind_list) > 2:
            num_episodes = min(len(task_ind_list), num_episodes_per_device)
            np.random.shuffle(task_ind_list)
            task_ind_list = task_ind_list[:num_episodes]

            task_list_one_device = []
            for tid in task_ind_list:
                sid_1 = task_support_list[tid].pop()
                sid_2 = task_support_list[tid].pop()
                support_1 = dataset[self.TEST_DATASET_LIST[tid]]['train'][f'{sid_1}_{num_shots}']
                support_2 = dataset[self.TEST_DATASET_LIST[tid]]['train'][f'{sid_2}_{num_shots}']
                if support_1[0].shape[0] > 0 and support_2[0].shape[0] > 0:
                    label_feature = dataset[self.TEST_DATASET_LIST[tid]]['label_features']
                    labels = dataset[self.TEST_DATASET_LIST[tid]]['label_list']
                    episode = self._prepare_episode(support_1, support_2, num_shots,
                                                    label_feature,
                                                    labels,
                                                    device,
                                                    reorder_query=True)
                    task_list_one_device.append(episode)
                    task_name_list.append(self.TEST_DATASET_LIST[tid])
            if len(task_list_one_device) > 1:
                yield MetaBatch({get_device_name(device): task_list_one_device}, task_name_list=task_name_list)
                task_name_list = []
                task_ind_list = list(range(len(self.TEST_DATASET_LIST)))
                task_ind_list = [t for t in task_support_list if len(task_support_list[t]) >= 2]

    def _prepare_episode(self, support, query, num_shots, label_features, text_labels, device=None, reorder_query=False):
        """ Batch -> Episode """
        if support[3].max() + 1 != support[3].unique().shape[0]:
            raise ValueError('Largest class id should match number of classes.')
       
        num_support = support[0].shape[0]
        num_classes = support[3].unique().shape[0]
        assert num_shots * num_classes == num_support

        def reorder(_set):
            reordered = []
            assert len(_set) == 4
            for i in range(len(_set)):
                reordered_item = torch.stack([_set[i][_set[3] == label] for label in _set[3].unique(sorted=True)])
                reordered_item = reordered_item.transpose(0, 1).reshape(-1, *reordered_item.shape[2:])
                reordered.append(reordered_item)
            return reordered
        support = reorder(support)

        if reorder_query:
            query = reorder(query)

        if device is not None:
            support = tuple(t.to(device) for t in support)
            query = tuple(t.to(device) for t in query)
        
    
        num_shots_per_batch = 4
        if num_shots % num_shots_per_batch != 0:
            raise ValueError(f'num_support_batches = num_shots / num_shots_per_batch should be an integer.')
        if num_shots // num_shots_per_batch == 1:
            # if the support set is too small, decrease the num of shots per batch
            num_shots_per_batch = 2

        return {'input_ids': torch.cat((support[0], query[0]), dim=0),
                'attention_mask': torch.cat((support[1], query[1]), dim=0),
                'token_type_ids': torch.cat((support[2], query[2]), dim=0),
                'labels': torch.cat((support[3], query[3]), dim=0),
                'num_shots': torch.tensor(num_shots).to(device),
                'num_support_batches': torch.tensor(num_shots // num_shots_per_batch).to(device),
                'num_classes': torch.tensor(num_classes).to(device),
                'label_features': tuple(t.to(device) for t in label_features),
                'text_labels': text_labels
                }

    def train_task_gen(self, num_shots):
        if num_shots not in self.NUM_SHOTS_LIST:
            raise ValueError(f'Only numbers of shots in {self.NUM_SHOTS_LIST} are supported')
        
        for i in self.TRAIN_SET_INDEX_LIST:
            yield self.dataset['train'][f'{i}_{num_shots}']
        
    def get_test_dataset_list(self):
        return self.TEST_DATASET_LIST

    def get_val_dataset_list(self):
        return self.VAL_DATASET_LIST
    
    def get_train_examples(self, dataset, index=0, num_shots=4):
        file_path = os.path.join(self.data_dir, dataset, f'{dataset}_train_{index}_{num_shots}.json')
        return self._create_examples(json.load(open(file_path, 'r')),
                                     f'train-{index}-{num_shots}')
    def get_test_examples(self, dataset):
        file_path = os.path.join(self.data_dir, dataset, f'{dataset}_eval.json')
        return self._create_examples(json.load(open(file_path, 'r')), 'test')

    def get_labels(self, dataset):
        file_path = os.path.join(self.data_dir, dataset, f'{dataset}_eval.json')
        return list(set([data['label'] for data in json.load(open(file_path, 'r'))]))

    def _create_examples(self, data_list, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data) in enumerate(data_list):
            guid = "%s-%d" % (set_type, i)
            text_a = data['sentence1']
            if 'sentence2' in data:
                text_b = data['sentence2']
            else:
                text_b = None
            label = data['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def examples_to_features(self, examples, label_list):
        return glue_convert_examples_to_features(examples,
                                                self.tokenizer,
                                                label_list=label_list,
                                                max_length=self.args.max_seq_length,
                                                output_mode="classification",
                                                pad_on_left=False,
                                                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                                                pad_token_segment_id=0)
    
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
        

    def load_and_cache_examples(self, validation=False):
        """[summary]

        Args:
            args ([type]): [description]
            tokenizer ([type]): [description]
            validation (bool, optional): Whether to load the validation datasets. Defaults to False.
            load_train_set (bool, optional): Whether to load the train sets. If False, load the test set.
                Defaults to True.
        """
        task_list = self.get_val_dataset_list() if validation else self.get_test_dataset_list()
        dataset_dict = {}
        for task in task_list:
            # Load data features from cache or dataset file
            cached_fname = 'cached_{}_{}_{}'.format(task, self.args.lm_type, str(self.args.max_seq_length))
            cached_features_file = os.path.join(self.data_dir, cached_fname)
            if os.path.exists(cached_features_file) and not self.args.overwrite_cache:
                logger.info("Loading features from cached file %s", cached_features_file)
                features = torch.load(cached_features_file)
                logger.info(f"number of test examples is {len(features['test'])}")
            else:
                logger.info("Creating features of %s", task)
                label_list = self.get_labels(task)
                # TODO: some datasets use number as labels
                label_features = self.text_to_features(' # '.join([str(l) for l in label_list]), task_list.index(task))

                train_features_dict = {}
                for index in self.TRAIN_SET_INDEX_LIST:
                    for num_shots in self.NUM_SHOTS_LIST:
                        examples = self.get_train_examples(task, index, num_shots)
                        features = self.examples_to_features(examples, label_list)
                        train_features_dict[f'{index}_{num_shots}'] = features

                test_examples = self.get_test_examples(task)
                test_features = self.examples_to_features(test_examples, label_list)

                logger.info("Saving features into cached file %s", cached_features_file)
                features = {
                    'train': train_features_dict,
                    'test': test_features,
                    'label_list': label_list,
                    'label_features': label_features
                }
                torch.save(features, cached_features_file)
            # conver to datasets 
            dataset_dict[task] = {}
            dataset_dict[task]['test'] = self.features_to_tensors(features['test'])
            dataset_dict[task]['train'] = {k: self.features_to_tensors(v) for k,v in features['train'].items()}
            dataset_dict[task]['label_list'] = features['label_list']
            dataset_dict[task]['label_features'] = self.features_to_tensors([features['label_features']])
        return dataset_dict
    
    def features_to_tensors(self, features):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        return (all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    
class LeopardDataProcessor_0526(LeopardDataProcessor):
    TEST_DATASET_LIST = ['airline', 'emotion', 'political_message',
                        'rating_electronics', 'political_audience',
                        'rating_books', 'rating_kitchen', 'sentiment_books',
                        'sentiment_kitchen', 'disaster', 'political_bias',
                        'rating_dvd', 'restaurant', 'sentiment_dvd' ]
    TEST_DATASET_LIST += ['huffpost_10', 'yelp', 'snips'] 
    VAL_DATASET_LIST = ['scitail']

    def __init__(self, args, tokenizer, data_dir):
        super().__init__(args, tokenizer, data_dir)