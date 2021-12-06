""" Baselines """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import os
import tqdm
import numpy as np
from collections import OrderedDict
from itertools import chain
from threading import local
from numpy.testing._private.utils import requires_memory
from requests.api import get

import torch
from numpy import inner
from sklearn.metrics import roc_auc_score
from torch import adaptive_avg_pool1d, device, dtype, nn
from torch import autograd
from torch.autograd import grad
from torch.autograd.grad_mode import no_grad
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
import torch.nn.functional as F

from logging_utils import get_logger
from modeling_adaptation import PrototypeBuildingNetwork
from modeling_bert_adaptation import BertLayerWithBNAdapter_AdaptContextShiftScaleAR, BertModelWithBNAdapter
from modeling_mixin import ClassificationAccMixin, CrossEntropyMixin, GetDeviceMixin
from transformers import AdamW
from modeling_bert import BertModel as BertForwardWithParams
from modeling_protonet import ProtoNetForTextClassificationWithBNAdapterEuclidean, ProtoNetForTextClassificationEuclidean, ProtoNetForTextClassificationWithBNAdapterEuclidean_FinetuneFilmOnly

from transformers.modeling_bert import BertModel as HuggingfaceBertModel

logger = get_logger('Baselines')

class BERTClassifier(nn.Module):
    def __init__(self, args, bert_name, bert_config, num_classes):
        super().__init__()
        self.bert = HuggingfaceBertModel.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.linear = nn.Linear(bert_config.hidden_size, num_classes)

    arg_list = ['input_ids', 'attention_mask', 'token_type_ids']
    def forward(self, batch, start=None, end=None):
        start = 0 if start is None else start
        end = batch['input_ids'].shape[0] if end is None else end
        assert start >=0 and start < end

        # Text encoding using BERT
        embs = self.bert(**{k:v[start:end] for k,v in batch.items() if k in self.arg_list})[1]
        embs = self.dropout(embs)
        embs = self.linear(embs)
        return embs

class BERTWithBNClassifier(BERTClassifier):
    def __init__(self, args, bert_name, bert_config, num_classes):
        super().__init__(args, bert_name, bert_config, num_classes)
        self.bert = BertModelWithBNAdapter.from_pretrained(
                bert_name,
                from_tf=False,
                config=bert_config,
                cache_dir=args.cache_dir if args.cache_dir else None)
        # Freeze the parameters of BERT
        for n, p in self.bert.named_parameters():
            if not 'adapter' in n and not 'LayerNorm' in n:
                p.requires_grad = False

class BERTWithFilmClassifier(BERTClassifier):
    def __init__(self, args, bert_name, bert_config, num_classes):
        super().__init__(args, bert_name, bert_config, num_classes)

class ProtoNetClassifier(nn.Module):
    def __init__(self, args, bert_name, bert_config, num_classes):
        super().__init__()
        self.protonet = ProtoNetForTextClassificationEuclidean(args, bert_name, bert_config)

    arg_list = ['input_ids', 'attention_mask', 'token_type_ids']
    def forward(self, batch, start=None, end=None):
        start = 0 if start is None else start
        end = batch['input_ids'].shape[0] if end is None else end
        assert start >=0 and start < end

        # Text encoding using BERT
        embs = self.bert(**{k:v[start:end] for k,v in batch.items() if k in self.arg_list})[1]
        embs = self.dropout(embs)
        embs = self.linear(embs)
        return embs

class ProtoNetWithBNClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class ProtoNetWithFilmClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class FineTuningBaseline(CrossEntropyMixin, ClassificationAccMixin,
                         GetDeviceMixin, nn.Module):
    """ Fine tuning on support set and test on query set.
        No meta-learning happens in this baseline.
    """
    def __init__(self, args, bert_name, bert_config):
        super().__init__()
        self.args = args
        self.bert_name = bert_name
        self.bert_config = bert_config
        self.classifier_class = BERTClassifier
        self.dummy_param = nn.Parameter(torch.ones(1))

    def forward(self, batch, eval=False, test_per_epoch=False):
        device = self.get_device()
        device_name = self.get_device_name()
        if hasattr(batch, device_name):
            local_batch = getattr(batch, device_name)
            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []

            for task in local_batch:
                if test_per_epoch:
                    trn_loss, trn_acc, tst_loss, tst_acc = self.eval_task(task, test_per_epoch=test_per_epoch)
                    train_loss.append(trn_loss)
                    train_acc.append(trn_acc)
                    test_loss.append(tst_loss)
                    test_acc.append(tst_acc)
                else:
                    tst_loss, tst_acc = self.eval_task(task, test_per_epoch=test_per_epoch)
                    test_loss.append(tst_loss)
                    test_acc.append(tst_acc)

            if test_per_epoch:
                return torch.stack(train_loss), torch.stack(train_acc), torch.stack(test_loss), torch.stack(test_acc)
            else:
                return torch.stack(test_loss), torch.stack(test_acc)
                
        else:
            assert eval # this could happen only during evaluation with multiple gpu
            loss = torch.ones(self.args.num_episodes_per_device) if not test_per_epoch else torch.ones(self.args.num_episodes_per_device, self.args.fine_tune_epochs)
            acc = torch.ones(self.args.num_episodes_per_device) if not test_per_epoch else torch.ones(self.args.num_episodes_per_device, self.args.fine_tune_epochs)
            loss *= -1
            acc *= -1
            return loss.to(device), acc.to(device), loss.to(device), acc.to(device)
    
    def init_model_and_optim(self, num_classes):
        bert_classifier = self.classifier_class(self.args, self.bert_name, self.bert_config, num_classes)
        bert_classifier.to(self.get_device())
        optimizer = AdamW(bert_classifier.parameters(), lr=self.args.learning_rate)

        if self.args.checkpoint_path.lower() != 'none':
            checkpoint = torch.load(os.path.join(self.args.checkpoint_path, "exp_checkpoint.pt"))
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return bert_classifier, optimizer
        
    def eval_task(self, task, test_per_epoch=False):
        task_labels = task['labels']
        if not 'num_classes' in task:
            num_classes = task_labels.unique().shape[0]
            num_shots = self.args.num_shots_support
        else:
            num_classes = task['num_classes'].item()
            num_shots = task['num_shots'].item()
        num_support = num_classes * num_shots

        # Creat BERT classifier and optimzier
        bert_classifier, optimizer = self.init_model_and_optim(num_classes)

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        # Fine tune
        for b in tqdm.tqdm(range(self.args.fine_tune_epochs), desc='Fine-tuning'):
            bert_classifier.train()
            # disturb data before each epoch
            support_set = {k:v[:num_support] for k,v in task.items() if k in bert_classifier.arg_list}
            num_batch = min(32, num_support)
            idx = torch.randperm(num_support)
            for arg in support_set:
                support_set[arg] = support_set[arg][idx]
            for i in range(0, num_support, num_batch):
                end = min(num_support, i+num_batch)
                support_logits = bert_classifier(task, start=i, end=end)
                loss = self.loss(support_logits, task_labels[i:end])
                loss /= support_logits.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bert_classifier.eval()
            with torch.no_grad():
                logits = []
                num_batch = 1000
                for i in range(0, num_support, num_batch):
                    end = min(num_support, i+num_batch)
                    support_logits = bert_classifier(task, start=i, end=end)
                    logits.append(support_logits)
                logits = torch.cat(logits, dim=0)
                acc = self.accuracy_fn(logits, task_labels[:num_support])
                loss = self.loss(logits, task_labels[:num_support])
                train_loss_list.append(loss)
                train_acc_list.append(acc)
            if test_per_epoch:
                loss, acc =  self.test(bert_classifier, task, num_support, task_labels)
                test_loss_list.append(loss)
                test_acc_list.append(acc)

        if not test_per_epoch:
            # Test after fine tuning
            return self.test(bert_classifier, task, num_support, task_labels)
        else:
            return torch.stack(train_loss_list), torch.stack(train_acc_list), \
                    torch.stack(test_loss_list), torch.stack(test_acc_list)

    def test(self, bert_classifier, task, num_support, task_labels):
        device = self.get_device()
        with torch.no_grad():
            logits = []
            batch_size = 500
            for i in range(num_support, task_labels.shape[0], batch_size):
                # num_query * hidden_size
                query_logits = bert_classifier(task, start=i, end=i+batch_size).cpu()
                logits.append(query_logits)
            query_logits = torch.cat(logits, dim=0)
            loss = self.loss(query_logits, task_labels[num_support:].cpu())
            acc = self.accuracy_fn(query_logits, task_labels[num_support:].cpu())

            return loss.to(device), acc.to(device)

class FineTuningProtoNet(FineTuningBaseline):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.protonet_class = ProtoNetForTextClassificationEuclidean
    
    def eval_task(self, task, test_per_epoch=False):
        task_labels = task['labels']
        if not 'num_classes' in task:
            num_classes = task_labels.unique().shape[0]
            num_shots = self.args.num_shots_support
        else:
            num_classes = task['num_classes'].item()
            num_shots = task['num_shots'].item()
        num_support = num_classes * num_shots

        # Creat BERT classifier and optimzier
        protonet = self.protonet_class(self.args, self.bert_name, self.bert_config)
        protonet.to(self.get_device())

        with torch.no_grad():
            support_features = protonet._text_encoding(task, start=0, end=num_support)
            # num_classes * hidden_size
            prototypes = protonet.prototype_build(support_features, task_labels[:num_support])
        classifier_weights = torch.nn.Parameter(prototypes, requires_grad=True).to(self.get_device())
        
        optimizer = AdamW([p for p in protonet.parameters()] + [classifier_weights], lr=self.args.learning_rate)

        if self.args.checkpoint_path.lower() != 'none':
            checkpoint = torch.load(os.path.join(self.args.checkpoint_path, "exp_checkpoint.pt"))
            protonet.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"Loaded ProtoNet from {self.args.checkpoint_path}")

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        # Fine tune
        for b in tqdm.tqdm(range(self.args.fine_tune_epochs), desc='Fine-tuning'):
            protonet.train()
            # disturb data before each epoch
            support_set = {k:v[:num_support] for k,v in task.items() if k in ['input_ids', 'attention_mask', 'token_type_ids']}
            num_batch = min(32, num_support)
            idx = torch.randperm(num_support)
            for arg in support_set:
                support_set[arg] = support_set[arg][idx]
            for i in range(0, num_support, num_batch):
                end = min(num_support, i+num_batch)
                support_features = protonet._text_encoding(task, start=i, end=end)
                support_logits = protonet.dist(support_features.unsqueeze(1).expand(-1, num_classes, -1), # (end-i) * num_classes * hidden_size
                                        classifier_weights.expand(end-i, -1, -1)) # (end-i) * num_classes * hidden_size
                loss = self.loss(support_logits, task_labels[i:end])
                loss /= support_logits.shape[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        protonet.eval()
        return self.test(protonet, classifier_weights, task, num_support, task_labels, num_classes)

    def test(self, protonet, classifier_weights, task, num_support, task_labels, num_classes):
        device = self.get_device()
        with torch.no_grad():
            logits = []
            batch_size = 500
            for i in range(num_support, task_labels.shape[0], batch_size):
                query_features = protonet._text_encoding(task, start=i, end=i+batch_size)
                num_query = query_features.shape[0]
                query_logits = protonet.dist(query_features.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                        classifier_weights.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size
                logits.append(query_logits.cpu())
            query_logits = torch.cat(logits, dim=0)
            loss = self.loss(query_logits, task_labels[num_support:].cpu())
            acc = self.accuracy_fn(query_logits, task_labels[num_support:].cpu())
            return loss.to(device), acc.to(device)
    
class FineTuningBERTWithBN(FineTuningBaseline):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.classifier_class = BERTWithBNClassifier

class FineTuningBERTWithFilm(FineTuningBaseline):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        raise NotImplementedError()

class FineTuningBERT(FineTuningBaseline):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.classifier_class = BERTClassifier

class FineTuningProtoNetBN(FineTuningProtoNet):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.protonet_class = ProtoNetForTextClassificationWithBNAdapterEuclidean

class FineTuningProtoNetFilm(FineTuningProtoNet):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.protonet_class = ProtoNetForTextClassificationWithBNAdapterEuclidean_FinetuneFilmOnly