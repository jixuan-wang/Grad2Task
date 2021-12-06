#####################################################
# Implementation of the MAML-based few-shot learner #
# https://arxiv.org/abs/1911.03863                  #
#####################################################

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from itertools import chain
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

from utils import MyDataParallel, loss, aggregate_accuracy
from modeling_mixin import ClassificationAccMixin, CrossEntropyMixin, GetDeviceMixin
from modeling_adaptation import ( PrototypeBuildingNetwork)

# Customized BERT implementation which supports forward pass with specified parameters
from modeling_bert import (BertModel)
from transformers.modeling_bert import BertModel as HuggingfaceBertModel

from logging_utils import get_logger
logger = get_logger('Leopard')

class LeopardForMetatraining(CrossEntropyMixin,
                             ClassificationAccMixin,
                             GetDeviceMixin,
                             nn.Module):
    """ BERT with the FiLM adapter layer for meta-training on various tasks. """
    def __init__(self, args, bert_name, bert_config, num_innper_epochs=1):
        super().__init__()
        self.args = args
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.num_inner_epochs = num_innper_epochs
        self.encoder = HuggingfaceBertModel.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        # Hyperparameters used by the Leopard paper:
        # https://arxiv.org/pdf/1911.03863.pdf
        hidden_dim = 256

        # Linear transformation after BERT encoding 
        self.linear = nn.Linear(bert_config.hidden_size, hidden_dim)
        self.label_linear = nn.Linear(bert_config.hidden_size, hidden_dim+1) # plus one counts for the bias

        # Meta learned learning rate
        self.learning_rates = nn.Parameter(torch.ones(bert_config.num_hidden_layers + 4) * args.learning_rate)
        # Mapping from parameter names to index of learning rate in self.learning_rates
        param_name_lr_index = {}
        for name, _ in self.encoder.named_parameters():
            if 'embeddings' in name:
                param_name_lr_index[name.replace('.', '_')] = -4
            elif 'pooler' in name:
                param_name_lr_index[name.replace('.', '_')] = -3
            else:
                layer_num = name.split('.')[2]
                param_name_lr_index[name.replace('.', '_')] = int(layer_num)
        param_name_lr_index['linear'] = -2
        param_name_lr_index['label_linear'] = -1
        self.param_name_lr_index = param_name_lr_index
        
        # BERT encoder used for inner loop update
        self.encoder_pi = BertModel(bert_config)
        # BERT parameters for inner loop update
        encoder_pi_params = OrderedDict({n.replace('.', '_'):nn.Parameter(p.data) for n,p in self.encoder.named_parameters()})
        self.encoder_pi_params = nn.ParameterDict(encoder_pi_params)
        # Linear layer for label embedding for inner loop update 
        label_linear_pi_params = OrderedDict({n:nn.Parameter(p.data) for n,p in self.label_linear.named_parameters()})
        self.label_linear_pi_params = nn.ParameterDict(label_linear_pi_params)
        # Linear after BERT encoding for inner loop update 
        linear_pi_params = OrderedDict({n:nn.Parameter(p.data) for n,p in self.linear.named_parameters()})
        self.linear_pi_params = nn.ParameterDict(linear_pi_params)
        # Learning rates 
        self.learning_rates_pi = nn.Parameter(torch.ones(bert_config.num_hidden_layers + 4) * args.learning_rate)

        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        self.prototype_build = PrototypeBuildingNetwork()
        self.cos = nn.CosineSimilarity(dim=-1)

    def meta_parameters(self):
        for p in chain(self.encoder.parameters(), self.linear.parameters(), self.label_linear.parameters(), [self.learning_rates]):
            yield p

    def meta_named_parameters(self):
        for n, p in chain(self.encoder.named_parameters(),
                          self.linear.named_parameters(),
                          self.label_linear.named_parameters(),
                          [('learning_rates', self.learning_rates)]):
            yield n,p

    def parameters_pi(self):
        for p in chain(self.encoder_pi_params.values(), self.linear_pi_params.values(), self.label_linear_pi_params.values(), [self.learning_rates_pi]):
            yield p

    def update_pi(self):
        """ Copy latest parameters from outer loop. """
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if type(self.encoder) is MyDataParallel:
                    self.encoder_pi_params[name.replace('.', '_').replace('module_', '')].data.copy_(param.data) 
                else:
                    self.encoder_pi_params[name.replace('.', '_')].data.copy_(param.data)
            for name, param in self.linear.named_parameters():
                self.linear_pi_params[name].data.copy_(param.data)
            for name, param in self.label_linear.named_parameters():
                self.label_linear_pi_params[name].data.copy_(param.data)
            self.learning_rates_pi.data.copy_(self.learning_rates.data)
    
    def inner_loops(self, task, batch_size, num_support, num_classes,
                    encoder_pi_params,
                    label_linear_pi_params,
                    linear_pi_params): 
        p_list = list(encoder_pi_params.values()) + list(label_linear_pi_params.values()) + list(linear_pi_params.values())
        lr_ind_list = [self.param_name_lr_index[n] for n in encoder_pi_params] + \
                        [self.param_name_lr_index['label_linear'] for _ in label_linear_pi_params] + \
                            [self.param_name_lr_index['linear'] for _ in linear_pi_params]

        for _ in range(self.num_inner_epochs):
            for start in range(batch_size, num_support, batch_size):
                args = {k:v[0:batch_size] for k,v in task.items() if k in self.bert_arg_list}
                args['params'] = encoder_pi_params
                support_bert_features = self.encoder_pi(**args)[1]
                support_embs = self.dropout(support_bert_features)
                support_embs = F.linear(support_embs, linear_pi_params['weight'], linear_pi_params['bias'])

                label_embs = self.dropout(support_bert_features)
                label_embs = F.linear(label_embs,
                                    label_linear_pi_params['weight'],
                                    label_linear_pi_params['bias'])

                classifier_weights = self.prototype_build(label_embs, task['labels'][:batch_size])
                assert num_classes == classifier_weights.shape[0]

                args = {k:v[start:start+batch_size] for k,v in task.items() if k in self.bert_arg_list}
                args['params'] = encoder_pi_params
                batch_features = self.encoder_pi(**args)[1]
                embs = self.dropout(batch_features)
                embs = F.linear(embs, linear_pi_params['weight'], linear_pi_params['bias'])

                logits = F.linear(embs, classifier_weights[:, :-1], classifier_weights[:, -1])
                
                loss = self.loss(logits, task['labels'][start:start+batch_size])
                acc = self.accuracy_fn(logits, task['labels'][start:start+batch_size])

                grads = autograd.grad(loss, p_list)
                for grad, param, lr_ind in zip(grads, p_list, lr_ind_list):
                    param -= self.learning_rates_pi[lr_ind] * grad

    def forward_task(self, task):
        self.update_pi()

        # if self.args.n_gpu < 2:
        if True:
            # the following parameters are leaf variable
            # so we need clone() here for inplace operation later
            encoder_pi_params = {n:p.clone() for n,p in self.encoder_pi_params.items()}
            linear_pi_params = {n:p.clone() for n,p in self.linear_pi_params.items()}
            label_linear_pi_params = {n:p.clone() for n,p in self.label_linear_pi_params.items()}
        """
        else:
            # with Dataparallel, the following parameters are not leaf variable anymore
            encoder_pi_params = self.encoder_pi_params
            linear_pi_params = self.linear_pi_params
            label_linear_pi_params = self.label_linear_pi_params
        """
            
        
        task_labels = task['labels']
        num_classes = task_labels.unique().shape[0]
        num_shots = self.args.num_shots_support
        num_support = num_classes * num_shots
        support_batch_size  = num_support // self.args.num_support_batches 
        
        self.inner_loops(task, support_batch_size, num_support, num_classes,
                         encoder_pi_params,
                         label_linear_pi_params,
                         linear_pi_params)

        args = {k:v[0:support_batch_size] for k,v in task.items() if k in self.bert_arg_list}
        args['params'] = encoder_pi_params
        support_bert_features = self.encoder_pi(**args)[1]
        support_embs = self.dropout(support_bert_features)
        support_embs = F.linear(support_embs, linear_pi_params['weight'], linear_pi_params['bias'])

        label_embs = self.dropout(support_bert_features)
        label_embs = F.linear(label_embs,
                            label_linear_pi_params['weight'],
                            label_linear_pi_params['bias'])

        classifier_weights = self.prototype_build(label_embs, task['labels'][:support_batch_size])
        assert num_classes == classifier_weights.shape[0]
        
        args = {k:v[num_support:] for k,v in task.items() if k in self.bert_arg_list}
        args['params'] = encoder_pi_params
        batch_features = self.encoder_pi(**args)[1]
        embs = self.dropout(batch_features)
        embs = F.linear(embs, linear_pi_params['weight'], linear_pi_params['bias'])

        logits = F.linear(embs, classifier_weights[:, :-1], classifier_weights[:, -1])
        
        loss = self.loss(logits, task['labels'][num_support:])
        acc = self.accuracy_fn(logits, task['labels'][num_support:])

        p_list = list(self.encoder_pi_params.values()) + list(self.linear_pi_params.values()) + list(self.label_linear_pi_params.values()) + [self.learning_rates_pi]
        grads = autograd.grad(loss, p_list)
        return loss, acc, grads

    def setup_dataparallel(self):
        self.encoder = MyDataParallel(self.encoder)

    def inner_loops_for_eval(self, task, batch_size, num_support, num_classes):
        p_list = list(self.encoder.module.parameters()) + list(self.label_linear.parameters()) + list(self.linear.parameters())

        lr_ind_list = [self.param_name_lr_index[n.replace('.', '_')] for n,_ in self.encoder.module.named_parameters()] + \
                        [self.param_name_lr_index['label_linear'] for _ in self.label_linear.parameters()] + \
                            [self.param_name_lr_index['linear'] for _ in self.linear.parameters()]

        for _ in range(self.num_inner_epochs):
            for start in range(batch_size, num_support, batch_size):
                args = {k:v[0:batch_size] for k,v in task.items() if k in self.bert_arg_list}
                support_bert_features = self.encoder(**args)[1]
                support_embs = self.dropout(support_bert_features)
                support_embs = self.linear(support_embs)

                # label_embs = self.dropout(support_bert_features) # No need for dropout in eval
                label_embs = self.label_linear(support_bert_features)

                classifier_weights = self.prototype_build(label_embs, task['labels'][:batch_size])
                assert num_classes == classifier_weights.shape[0]

                args = {k:v[start:start+batch_size] for k,v in task.items() if k in self.bert_arg_list}
                batch_features = self.encoder(**args)[1]
                # embs = self.dropout(batch_features)
                embs = self.linear(batch_features)

                logits = F.linear(embs, classifier_weights[:, :-1], classifier_weights[:, -1])
                
                loss = self.loss(logits, task['labels'][start:start+batch_size])
                acc = self.accuracy_fn(logits, task['labels'][start:start+batch_size])

                grads = autograd.grad(loss, p_list)
                for grad, param, lr_ind in zip(grads, p_list, lr_ind_list):
                    # param -= self.learning_rates_pi[lr_ind] * grad
                    param.data.add_(-self.learning_rates[lr_ind].data, grad)
                self.encoder.zero_grad()
                self.linear.zero_grad()
                self.label_linear.zero_grad()
                

    def eval_task(self, task):
        # self.update_pi()
        p_list = list(self.encoder.module.parameters()) + list(self.label_linear.parameters()) + list(self.linear.parameters())
        with torch.no_grad():
            p_backup_list = [p.clone() for p in p_list]

        # if self.args.n_gpu < 2:
        #     # without Dataparallel, the following parameters are leaf variable
        #     # so we need clone() here for inplace operation later
        # encoder_pi_params = {n:p.clone() for n,p in self.encoder_pi_params.items()}
        # linear_pi_params = {n:p.clone() for n,p in self.linear_pi_params.items()}
        # label_linear_pi_params = {n:p.clone() for n,p in self.label_linear_pi_params.items()}
        # else:
        #     # with Dataparallel, the following parameters are not leaf variable anymore
        #     encoder_pi_params = self.encoder_pi_params
        #     linear_pi_params = self.linear_pi_params
        #     label_linear_pi_params = self.label_linear_pi_params

        device = self.get_device()
        task_labels = task['labels']
        if not 'num_shots' in task:
            num_classes = task_labels.unique().shape[0]
            num_shots = self.args.num_shots_support
            num_support_batches = self.args.num_support_batches
        else:
            num_classes = task['num_classes'].item()
            num_shots = task['num_shots'].item()
            num_support_batches = task['num_support_batches'].item()

        num_support = num_classes * num_shots
        num_query = task_labels.shape[0] - num_support
        support_batch_size  = num_support // num_support_batches
        
        self.inner_loops_for_eval(task, support_batch_size, num_support, num_classes)

        with torch.no_grad(): 
            args = {k:v[0:support_batch_size] for k,v in task.items() if k in self.bert_arg_list}
            support_bert_features = self.encoder(**args)[1]
            # support_embs = self.dropout(support_bert_features)
            # support_embs = F.linear(support_embs, linear_pi_params['weight'], linear_pi_params['bias'])

            label_embs = self.dropout(support_bert_features)
            label_embs = self.label_linear(label_embs)

            classifier_weights = self.prototype_build(label_embs, task['labels'][:support_batch_size])
            assert num_classes == classifier_weights.shape[0]

            logits = []
            b_size = 500
            for i in range(num_support, task_labels.shape[0], b_size):
                args = {k:v[i:i+b_size] for k,v in task.items() if k in self.bert_arg_list}
                batch_features = self.encoder(**args)[1]
                embs = self.linear(batch_features)
                query_logits = F.linear(embs, classifier_weights[:, :-1], classifier_weights[:, -1])

                query_logits = query_logits.cpu()

                logits.append(query_logits)
            query_logits = torch.cat(logits, dim=0)
            loss = self.loss(query_logits, task_labels[num_support:].cpu())
            acc = self.accuracy_fn(query_logits, task_labels[num_support:].cpu())

            # Recover the models to the initial state
            for new_p, old_p in zip(p_list, p_backup_list):
                new_p.data.copy_(old_p.data)
            device = self.get_device()
            return loss.to(device), acc.to(device)

    def forward(self, batch, eval=False):
        device = self.get_device()
        device_name = self.get_device_name()
        if hasattr(batch, device_name):
            local_batch = getattr(batch, device_name)
            loss = 0
            query_acc = []
            grads_list = []

            for task in local_batch:
                if not eval:
                    t_loss, t_acc, grads = self.forward_task(task)
                    grads_list.append([g.unsqueeze(0) for g in grads])
                    t_loss = t_loss.detach()
                    t_acc = t_acc.detach()
                else:
                    t_loss, t_acc = self.eval_task(task)
                loss += t_loss
                query_acc.append(t_acc)
            query_acc = torch.stack(query_acc).detach()

            if not eval:
                if len(grads_list) > 1:
                    grads_list = [torch.cat(g, dim=0) for g in zip(*grads_list)]
                else:
                    grads_list = grads_list[0]

            return (loss, query_acc) if eval else (loss, query_acc, grads_list)
        else:
            assert eval # this could happen only during evaluation with multiple gpu
            loss, query_acc = torch.tensor(0.).to(device), torch.tensor([-1.]).to(device)
            return loss, query_acc

    def dummy_forward(self, batch):
        """ This method is purely used for update encoder/cl_adaptation_network
            by gradients of encoder_pi/cl_adaptation_network_pi, by a dummy forward
            and backward pass.
        """
        device = self.get_device()
        device_name = self.get_device_name()
        local_batch = getattr(batch, device_name)
        task = local_batch[0]
        task_labels = task['labels']
        num_classes = task_labels.unique().shape[0]

        args = {k:v[0:num_classes] for k,v in task.items() if k in self.bert_arg_list}
        features = self.encoder(**args)[1]

        embs = self.linear(features)
        label_embs = self.label_linear(features)

        loss = embs.reshape(-1).mean() + label_embs.reshape(-1).mean() + self.learning_rates.mean()
        return loss

    bert_arg_list = ['input_ids', 'attention_mask', 'token_type_ids']
    def _bert_text_encoding(self, batch, start=None, end=None, use_pi=True):
        encoder = self.encoder_pi if use_pi else self.encoder

        start = 0 if start is None else start
        end = batch['input_ids'].shape[0] if end is None else end
        assert start >=0 and start < end
        
        # Text encoding using BERT
        embs = encoder(**{k:v[start:end] for k,v in batch.items() if k in self.bert_arg_list})[1]
        return embs


class LeopardFixedLR(LeopardForMetatraining):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)

    def meta_parameters(self):
        for p in chain(self.encoder.parameters(), self.label_linear.parameters()):
            yield p

    def meta_named_parameters(self):
        for n, p in chain(self.encoder.named_parameters(),
                          self.label_linear.named_parameters()):
            yield n,p
    
    def inner_loops(self, task, batch_size, num_support, num_classes,
                    encoder_pi_params,
                    label_linear_pi_params,
                    linear_params): 
        p_list = list(encoder_pi_params.values()) + list(label_linear_pi_params.values()) + list(linear_params.values())
        lr_ind_list = [self.param_name_lr_index[n] for n in encoder_pi_params] + \
                        [self.param_name_lr_index['label_linear'] for _ in label_linear_pi_params] + \
                            [self.param_name_lr_index['linear'] for _ in linear_params]

        for start in range(batch_size, num_support, batch_size):
            args = {k:v[0:batch_size] for k,v in task.items() if k in self.bert_arg_list}
            args['params'] = encoder_pi_params
            support_bert_features = self.encoder_pi(**args)[1]
            support_embs = self.dropout(support_bert_features)
            support_embs = F.linear(support_embs, linear_params['weight'], linear_params['bias'])

            label_embs = self.dropout(support_bert_features)
            label_embs = F.linear(label_embs,
                                label_linear_pi_params['weight'],
                                label_linear_pi_params['bias'])

            classifier_weights = self.prototype_build(label_embs, task['labels'][:batch_size])
            assert num_classes == classifier_weights.shape[0]

            args = {k:v[start:start+batch_size] for k,v in task.items() if k in self.bert_arg_list}
            args['params'] = encoder_pi_params
            batch_features = self.encoder_pi(**args)[1]
            embs = self.dropout(batch_features)
            embs = F.linear(embs, linear_params['weight'], linear_params['bias'])

            logits = F.linear(embs, classifier_weights[:, :-1], classifier_weights[:, -1])
            
            loss = self.loss(logits, task['labels'][start:start+batch_size])
            acc = self.accuracy_fn(logits, task['labels'][start:start+batch_size])

            grads = autograd.grad(loss, p_list)
            for grad, param, lr_ind in zip(grads, p_list, lr_ind_list):
                param -= 0.0001 * grad

    def forward_task(self, task):
        self.update_pi()

        if self.args.n_gpu < 2:
            # without Dataparallel, the following parameters are leaf variable
            # so we need clone() here for inplace operation later
            encoder_pi_params = {n:p.clone() for n,p in self.encoder_pi_params.items()}
            linear_params = {n:p.clone() for n,p in self.linear_params.items()}
            label_linear_pi_params = {n:p.clone() for n,p in self.label_linear_pi_params.items()}
        else:
            # with Dataparallel, the following parameters are not leaf variable anymore
            encoder_pi_params = self.encoder_pi_params
            linear_params = self.linear_params
            label_linear_pi_params = self.label_linear_pi_params
            
        
        task_labels = task['labels']
        num_classes = task_labels.unique().shape[0]
        num_shots = self.args.num_shots_support
        num_support = num_classes * num_shots
        support_batch_size  = num_support // self.args.num_support_batches 
        
        self.inner_loops(task, support_batch_size, num_support, num_classes,
                         encoder_pi_params,
                         label_linear_pi_params,
                         linear_params)

        args = {k:v[0:support_batch_size] for k,v in task.items() if k in self.bert_arg_list}
        args['params'] = encoder_pi_params
        support_bert_features = self.encoder_pi(**args)[1]
        support_embs = self.dropout(support_bert_features)
        support_embs = F.linear(support_embs, linear_params['weight'], linear_params['bias'])

        label_embs = self.dropout(support_bert_features)
        label_embs = F.linear(label_embs,
                            label_linear_pi_params['weight'],
                            label_linear_pi_params['bias'])

        classifier_weights = self.prototype_build(label_embs, task['labels'][:support_batch_size])
        assert num_classes == classifier_weights.shape[0]
        
        args = {k:v[num_support:] for k,v in task.items() if k in self.bert_arg_list}
        args['params'] = encoder_pi_params
        batch_features = self.encoder_pi(**args)[1]
        embs = self.dropout(batch_features)
        embs = F.linear(embs, linear_params['weight'], linear_params['bias'])

        logits = F.linear(embs, classifier_weights[:, :-1], classifier_weights[:, -1])
        
        loss = self.loss(logits, task['labels'][num_support:])
        acc = self.accuracy_fn(logits, task['labels'][num_support:])

        p_list = list(self.encoder_pi_params.values()) + list(self.label_linear_pi_params.values())
        grads = autograd.grad(loss, p_list)
        return loss, acc, grads
