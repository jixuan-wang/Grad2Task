from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os
from collections import OrderedDict
from itertools import chain
from threading import local

import torch
from numpy import inner
from torch import nn
from torch.autograd.grad_mode import no_grad

from logging_utils import get_logger
from modeling_adaptation import PrototypeBuildingNetwork
from modeling_bert import BertModel
from modeling_bert_adaptation import BertModelWithBNAdapter, BertModelWithBNAdapter_WithoutTaskAdaptationWithShiftSacle
from modeling_mixin import ClassificationAccMixin, CrossEntropyMixin, GetDeviceMixin
from transformers import AdamW
from transformers.modeling_bert import BertModel as HuggingfaceBertModel
from utils import EuclideanDist

logger = get_logger('ProtoNet')

class ProtoNetForTextClassification(CrossEntropyMixin,
                                    ClassificationAccMixin,
                                    GetDeviceMixin,
                                    nn.Module):
    """ ProtoNet for diverse text classification tasks. """

    def __init__(self, args, bert_name, bert_config):
        super().__init__()
        self.args = args
        self.bert = HuggingfaceBertModel.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)

        self.prototype_build = PrototypeBuildingNetwork()
        self.dist = nn.CosineSimilarity(dim=-1)

    def forward(self, batch, eval=False):
        device = self.get_device()
        device_name = self.get_device_name()
        if hasattr(batch, device_name):
            local_batch = getattr(batch, device_name)
            loss = 0
            query_acc = []

            for task in local_batch:
                if not eval:
                    t_loss, t_acc = self.forward_task(task)
                else:
                    t_loss, t_acc = self.eval_task(task)
                loss += t_loss
                query_acc.append(t_acc)
            query_acc = torch.stack(query_acc)
            return loss, query_acc
        else:
            assert eval # this could happen only during evaluation with multiple gpu
            loss, query_acc = torch.tensor(0.).to(device), torch.tensor([-1.]).to(device)
            return loss, query_acc

    def forward_task(self, task):
        task_labels = task['labels']
        num_classes = task_labels.unique().shape[0]
        num_shots = self.args.num_shots_support
        num_support = num_classes * num_shots

        task_features = self._text_encoding(task)
        # num_classes * hidden_size
        prototypes = self.prototype_build(task_features[:num_support],
                                        task_labels[:num_support])
        assert num_classes == prototypes.shape[0]
        # num_query * hidden_size
        query_features = task_features[num_support:]
        num_query = query_features.shape[0]

        # num_query * num_classes
        query_logits = self.dist(query_features.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

        loss = self.loss(query_logits, task_labels[num_support:])
        acc = self.accuracy_fn(query_logits, task_labels[num_support:])

        return loss, acc

    def eval_task(self, task):
        device = self.get_device()
        task_labels = task['labels']
        if not 'num_classes' in task:
            num_classes = task_labels.unique().shape[0]
            num_shots = self.args.num_shots_support
        else:
            num_classes = task['num_classes'].item()
            num_shots = task['num_shots'].item()
        num_support = num_classes * num_shots

        with torch.no_grad():
            support_features = self._text_encoding(task, start=0, end=num_support)
            # num_classes * hidden_size
            prototypes = self.prototype_build(support_features, task_labels[:num_support])
            prototypes = prototypes.cpu()
            assert num_classes == prototypes.shape[0]

            logits = []
            batch_size = 50
            for i in range(num_support, task_labels.shape[0], batch_size):
                # num_query * hidden_size
                query_features = self._text_encoding(task, start=i, end=i+batch_size)
                query_features = query_features.cpu()
                num_query = query_features.shape[0]

                # num_query * num_classes
                query_logits = self.dist(query_features.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                        prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size
                logits.append(query_logits)
            query_logits = torch.cat(logits, dim=0)
            loss = self.loss(query_logits, task_labels[num_support:].cpu())
            acc = self.accuracy_fn(query_logits, task_labels[num_support:].cpu())

            return loss.to(device), acc.to(device)

    def _text_encoding(self, batch, params=None, start=None, end=None):
        arg_list = ['input_ids', 'attention_mask', 'token_type_ids']

        start = 0 if start is None else start
        end = batch['input_ids'].shape[0] if end is None else end
        assert start >=0 and start < end
        
        # Text encoding using BERT
        if params is None:
            embs = self.bert(**{k:v[start:end] for k,v in batch.items() if k in arg_list})[1]
        else:
            embs = self.bert(**{k:v[start:end] for k,v in batch.items() if k in arg_list}, params=params)[1]
        embs = self.dropout(embs)
        embs = self.linear(embs)
        return embs

class ProtoNetForTextClassificationWithBNAdapter(ProtoNetForTextClassification):
    """ ProtoNet for diverse text classification tasks. """
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        
        self.bert = BertModelWithBNAdapter.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        # Freeze the parameters of BERT
        for n, p in self.bert.named_parameters():
            if not 'adapter' in n and not 'LayerNorm' in n:
                p.requires_grad = False

class ProtoNetForTextClassificationWithLayerNormAdaptation(ProtoNetForTextClassification):
    """ ProtoNet for diverse text classification tasks.
        Only layer norm layers are updated in BERT.
    """
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        
        # Freeze the parameters of BERT except layer norm layers 
        for n, p in self.bert.named_parameters():
            if not 'LayerNorm' in n:
                p.requires_grad = False

class ProtoNetForTextClassificationEuclidean(ProtoNetForTextClassification):
    """ ProtoNet for diverse text classification tasks. """
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.dist = EuclideanDist(dim=-1)
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)
        
class ProtoNetForTextClassificationWithBNAdapterEuclidean(ProtoNetForTextClassification):
    """ ProtoNet for diverse text classification tasks. """
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        
        self.bert = BertModelWithBNAdapter.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        # Freeze the parameters of BERT
        for n, p in self.bert.named_parameters():
            if not 'adapter' in n and not 'LayerNorm' in n:
                p.requires_grad = False

        self.dist = EuclideanDist(dim=-1)
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)

class ProtoNetForTextClassificationWithBNAdapterEuclidean_FinetuneFilmOnly(ProtoNetForTextClassification):
    """ ProtoNet for diverse text classification tasks. """
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        
        self.bert = BertModelWithBNAdapter_WithoutTaskAdaptationWithShiftSacle.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        # Freeze the parameters of BERT
        for n, p in self.bert.named_parameters():
            if not 'film' in n and not 'LayerNorm' in n:
                p.requires_grad = False

        self.dist = EuclideanDist(dim=-1)
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)

class ProtoNetForTextClassificationWithLayerNormAdaptationEuclidean(ProtoNetForTextClassification):
    """ ProtoNet for diverse text classification tasks.
        Only layer norm layers are updated in BERT.
    """
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        
        # Freeze the parameters of BERT except layer norm layers 
        for n, p in self.bert.named_parameters():
            if not 'LayerNorm' in n:
                p.requires_grad = False

        self.dist = EuclideanDist(dim=-1)
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)