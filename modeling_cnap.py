#####################################################
# Variants of Conditional Neural Adaptive Processes #
# https://arxiv.org/abs/1906.07697                  #
# e.g., using gradients as task representation,     #
# using input encoding as task representation, etc. #
#####################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from logging_utils import get_logger
from modeling_adaptation import PrototypeBuildingNetwork
from modeling_bert_adaptation import (
    BertModelWithBNAdapter, BertModelWithBNAdapter_AdaptContextShiftScaleAR,
    BertModelWithBNAdapterAdaptContextTaskEmb,
    BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScale,
    BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR,
    BertModelWithBNAdapterAdaptInputOutput,
    BertModelWithBNAdapterAdaptInputOutputTaskEmb)
from modeling_mixin import (ClassificationAccMixin, CrossEntropyMixin,
                            GetDeviceMixin)
from transformers import AdamW
from transformers.modeling_bert import BertModel as HuggingfaceBertModel
from utils import EuclideanDist, dist_metric_by_name

logger = get_logger('CNAP')

class LoadPretrainedBaseMixIn:
    def load_pretrained_base_model(self):
        if hasattr(self, 'pretrained_state_dict'):
            # Loading from pretrained BERT with bottleneck layer models 
            state_dict = self.state_dict()
            for pt_n, pt_p in self.pretrained_state_dict.items():
                if pt_n in state_dict: 
                    state_dict[pt_n].copy_(pt_p.data)
        else:
            raise ValueError('pretrained_state_dict need to be specified.')

class InitAdaptForHiddenMixIn:
    def init_adapt_module_dict(self):
        adapt_module_dict = {}
        for i in range(self.bert_config.num_hidden_layers):
            adapt_module_dict[f'{i}_{0}_scale'] = nn.Linear(self.args.task_emb_size, self.bert_config.bn_adapter_hidden_size)
            adapt_module_dict[f'{i}_{0}_shift'] = nn.Linear(self.args.task_emb_size, self.bert_config.bn_adapter_hidden_size)
            adapt_module_dict[f'{i}_{1}_scale'] = nn.Linear(self.args.task_emb_size, self.bert_config.bn_adapter_hidden_size)
            adapt_module_dict[f'{i}_{1}_shift'] = nn.Linear(self.args.task_emb_size, self.bert_config.bn_adapter_hidden_size)
        return nn.ModuleDict(adapt_module_dict)

class InitAdaptForInputOutputMixIn:
    def init_adapt_module_dict(self):
        adapt_module_dict = {}
        for i in range(self.bert_config.num_hidden_layers):
            adapt_module_dict[f'{i}_attention_scale_input'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_attention_scale_output'] = nn.Linear(self.args.task_emb_size,self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_attention_shift_input'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_attention_shift_output'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_output_scale_input'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_output_scale_output'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_output_shift_input'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
            adapt_module_dict[f'{i}_output_shift_output'] = nn.Linear(self.args.task_emb_size, self.bert_config.hidden_size)
        return nn.ModuleDict(adapt_module_dict)
    

class GenShiftScaleParamsForHiddenMixIn:
    def _shift_scale_params(self, task_emb):
        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({
                'attention': {'scale': torch.tanh(self.adapt_module_dict[f'{i}_{0}_scale'](task_emb)).reshape(-1) + 1,
                              'shift': self.adapt_module_dict[f'{i}_{0}_shift'](task_emb).reshape(-1)},
                'output': {'scale': torch.tanh(self.adapt_module_dict[f'{i}_{1}_scale'](task_emb)).reshape(-1) + 1,
                           'shift': self.adapt_module_dict[f'{i}_{1}_shift'](task_emb).reshape(-1)}
            })
        return param_dict_list

class GenShiftScaleParamsForInputOutputMixIn:
    def _shift_scale_params(self, task_emb):
        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({
                'attention': {'scale_in': torch.tanh(self.adapt_module_dict[f'{i}_attention_scale_input'](task_emb[i:i+1])).reshape(-1) + 1,
                              'scale_out': torch.tanh(self.adapt_module_dict[f'{i}_attention_scale_output'](task_emb[i:i+1])).reshape(-1) + 1,
                              'shift_in': self.adapt_module_dict[f'{i}_attention_shift_input'](task_emb[i:i+1]).reshape(-1),
                              'shift_out': self.adapt_module_dict[f'{i}_attention_shift_output'](task_emb[i:i+1]).reshape(-1)},
                'output': {'scale_in': torch.tanh(self.adapt_module_dict[f'{i}_output_scale_input'](task_emb[i:i+1])).reshape(-1) + 1,
                           'scale_out': torch.tanh(self.adapt_module_dict[f'{i}_output_scale_output'](task_emb[i:i+1])).reshape(-1) + 1,
                           'shift_in': self.adapt_module_dict[f'{i}_output_shift_input'](task_emb[i:i+1]).reshape(-1),
                           'shift_out': self.adapt_module_dict[f'{i}_output_shift_output'](task_emb[i:i+1]).reshape(-1)}
            })
        return param_dict_list

class GenTaskEmbForShiftScaleMixIn:
    def _shift_scale_params(self, task_emb):
        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({ 'attention': {'task_emb': task_emb[i]}, 
                                     'output': {'task_emb': task_emb[i]}})
        return param_dict_list

class GenTaskEmbForEachAdapterMixIn:
    def _shift_scale_params(self, task_emb):
        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({ 'attention': {'task_emb': task_emb[i*2]}, 
                                     'output': {'task_emb': task_emb[i*2+1]}})
        return param_dict_list

"""
class GenTaskEmbForShiftScaleFineGrainedMixIn:
    def _shift_scale_params(self, task_emb):
        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({ 'attention': {'task_emb': task_emb[i:i+4]}, 
                                     'output': {'task_emb': task_emb[i:i+4]}})
        return param_dict_list
"""

class CNAP(CrossEntropyMixin, ClassificationAccMixin, GetDeviceMixin, nn.Module):
    """ Base class for CNAP.  """

    bert_arg_list = ['input_ids', 'attention_mask', 'token_type_ids']
    batch_arg_list = bert_arg_list + ['labels']

    def __init__(self, args, bert_name, bert_config):
        super().__init__()
        self.args = args
        self.pretrained_bert_name = bert_name
        self.bert_config = bert_config
        self.set_bert_type()
        self.bert = self.init_bert()
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # linear layer after BERT
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)

        self.setup_grad()

        self.prototype_build = PrototypeBuildingNetwork()
        # self.cos = nn.CosineSimilarity(dim=-1)
        self.dist_metric = self.init_dist_metric()
        # Task embedding model
        self.task_emb_model = self.init_task_emb_model()
        # Adaptation network 
        self.adapt_module_dict = self.init_adapt_module_dict()


    def init_task_emb_model(self):
        raise NotImplementedError()
    def init_adapt_module_dict(self):
        logger.warning('No Adaptation network is provided.')

    def set_bert_type(self):
        # Set bert_type here
        raise NotImplementedError()

    def init_dist_metric(self):
        return EuclideanDist(dim=-1)

    def init_bert(self):
        return self.bert_class.from_pretrained(
                        self.pretrained_bert_name,
                        from_tf=False,
                        config=self.bert_config,
                        cache_dir=self.args.cache_dir if self.args.cache_dir else None)

    def setup_grad(self):
        if self.args.cnap_freeze_base_model:
            # Freenze the BERT model with linear.
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            # Only bottleneck adapters and last linear layer are trainable.
            for n, p in self.bert.named_parameters():
                if not 'adapter' in n:
                    p.requires_grad = False

        if self.args.cnap_freeze_linear_layer:
            # Whether to freeze the last linear layer.
            for p in self.linear.parameters():
                p.requires_grad = False

    def _text_encoding_by_batch(self, task, add_arg_names, add_arg_values, batch_size, cpu=False):
        features_list = []
        total_num = task['input_ids'].shape[0]
        for i in range(0, total_num, batch_size):
            features = self._text_encoding(task, start=i, end=i+batch_size,
                                           add_arg_names = add_arg_names,
                                           add_arg_values = add_arg_values)
            features = features.cpu() if cpu else features
            features_list.append(features)
        return torch.cat(features_list)

    def _text_encoding(self, batch, start=None, end=None, add_arg_names=None, add_arg_values=None):
        start = 0 if start is None else start
        end = batch['input_ids'].shape[0] if end is None else end
        if not (start >=0 and start < end):
            raise ValueError(f'Invalid start and end value: start {start}, end {end}')
        
        # Text encoding using BERT
        args = {k:v[start:end] for k,v in batch.items() if k in self.bert_arg_list}
        if add_arg_names is not None and add_arg_values is not None:
            assert len(add_arg_values) == len(add_arg_names)
            for n, v in zip(add_arg_names, add_arg_values):
                args[n] = v
        embs = self.bert(**args)[1]
        embs = self.dropout(embs)
        embs = self.linear(embs)
        return embs

    def protonet_loss_acc(self, spt_features, spt_labels, qry_features, qry_labels):
        # num_classes * hidden_size
        prototypes = self.prototype_build(spt_features, spt_labels)
        num_classes = spt_labels.unique().shape[0]
        assert num_classes == prototypes.shape[0]
        num_query = qry_features.shape[0]
        # num_query * num_classes
        qry_logits = self.dist_metric(
            qry_features.unsqueeze(1).expand(-1, num_classes, -1),  # num_query * num_classes * hidden_size
            prototypes.expand(num_query, -1, -1))                   # num_query * num_classes * hidden_size

        loss = self.loss(qry_logits, qry_labels)
        acc = self.accuracy_fn(qry_logits, qry_labels)
        return loss, acc

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
            return loss, query_acc.detach()
        else:
            assert eval # this could happen only during evaluation with multiple gpu
            loss, query_acc = torch.tensor(0.).to(device), torch.tensor([-1.]).to(device)
            return loss, query_acc

    def forward_task(self, task):
        task_labels = task['labels']
        num_classes = task_labels.unique().shape[0]
        num_shots = self.args.num_shots_support
        num_support = num_classes * num_shots

        adapt_arg_names, adapt_arg_values = self.gen_adapt_param(task, num_support,
                                                                 batch_size=num_support,
                                                                 detach=False)
        task_features = self._text_encoding(task, 
                                            add_arg_names = adapt_arg_names,
                                            add_arg_values = adapt_arg_values)

        # Prototypical loss.
        if not self.args.use_improve_loss:
            loss, acc = self.protonet_loss_acc(task_features[:num_support],
                                            task_labels[:num_support],
                                            task_features[num_support:],
                                            task_labels[num_support:])
        else:
            loss_after_adp, acc_after_adp = self.protonet_loss_acc(task_features[:num_support],
                                            task_labels[:num_support],
                                            task_features[num_support:],
                                            task_labels[num_support:])

            task_features = self._text_encoding(task)
            loss_before_adp, acc_before_adp = self.protonet_loss_acc(task_features[:num_support],
                                                                    task_labels[:num_support],
                                                                    task_features[num_support:],
                                                                    task_labels[num_support:])
            loss = F.relu(loss_after_adp - loss_before_adp + 1.0)
            import ipdb; ipdb.set_trace()
            acc = acc_after_adp
            
        return loss, acc

    def gen_adapt_param(self, task, num_support, batch_size, detach=False):
        if self.args.cnap_adapt:
            task_emb_sum = None
            task_emb_num = 0
            for ind in range(0, num_support, batch_size):
                task_emb = self._task_encoding({
                    k:v[ind:ind+batch_size] if k in self.batch_arg_list else v for k,v in task.items()})
                task_emb_num += 1
                task_emb_sum = task_emb if task_emb_sum is None else task_emb_sum + task_emb
            task_emb = task_emb_sum / task_emb_num 
            if detach:
                task_emb = task_emb.detach()
            adapt_param_list = self._shift_scale_params(task_emb)
            return ['params_dict_list'], [adapt_param_list]
        else:
            return None, None

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
        num_query = task_labels.shape[0] - num_support

        adapt_arg_names, adapt_arg_values = self.gen_adapt_param(task, num_support,
                                                                 num_support if num_shots <= 6 else 4*num_classes, #TODO: Good batch size?
                                                                 detach=True)
        with torch.no_grad():
            support_features = self._text_encoding_by_batch(
                {k:v[:num_support] if k in self.bert_arg_list else v for k,v in task.items()},
                adapt_arg_names,
                adapt_arg_values,
                num_support if num_shots <= 6 else 100,
                cpu=True
            )
            query_features = self._text_encoding_by_batch(
                {k:v[num_support:] if k in self.bert_arg_list else v for k,v in task.items()},
                adapt_arg_names,
                adapt_arg_values,
                min(100, num_query),
                cpu=True
            )

            loss, acc = self.protonet_loss_acc(support_features,
                                               task_labels[:num_support].cpu(),
                                               query_features,
                                               task_labels[num_support:].cpu())
            return loss.to(device), acc.to(device)

class CNAPWithBNAdapter(LoadPretrainedBaseMixIn,
                        InitAdaptForHiddenMixIn,
                        GenShiftScaleParamsForHiddenMixIn,
                        CNAP):
    """ CNAP with Fisher Information Matrix (FIM) as task representation. FIM is
        calculated by the gradients with regard to a prototypical loss on the 
        training data (support set) of a task.

        The task embedding model is a RNN model that, at each layer, outputs
        adaptation parameters given the FIM of the current layer and hidden 
        states until the current step.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict=None):
        super().__init__(args, bert_name, bert_config)

        self.pretrained_state_dict = pt_bert_bn_state_dict
        self.load_pretrained_base_model()

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapter

    def init_dist_metric(self):
        return nn.CosineSimilarity(dim=-1)

    def init_task_emb_model(self):
        return nn.RNN(16*768*4, self.args.task_emb_size, 2, nonlinearity='tanh', batch_first=True)

    def _task_encoding(self, task, num_steps=5):
        for n, p in self.bert.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        
        num_classes = task['labels'].unique().shape[0]
        num_examples = task['input_ids'].shape[0]
        num_support = num_examples // 2
       
        for i in range(num_steps):
            embs = self.bert(**task)[1]
            # embs = self.dropout(embs)
            embs = self.linear(embs)

            prototypes = self.prototype_build(embs[:num_support],
                                              task['labels'][:num_support])
            assert num_classes == prototypes.shape[0]
            query_embs = embs[num_support:]
            num_query = query_embs.shape[0]
            query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                    prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

            target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
            loss = self.loss(query_logits, target)
            # loss = self.loss(query_logits, task['labels'][num_support:])
            # acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
            self.bert.zero_grad()
            loss.backward()
            for n, p in self.bert.named_parameters():
                if 'adapter' in n:
                    assert p.grad is not None
                    p.grad_accumu += p.grad.data ** 2 
                    p.grad_accumu_count += 1
        
        for n,p in self.bert.named_parameters():
            if 'adapter' in n:
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False
        
        all_grads = [] 
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))
        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(self.bert.config.num_hidden_layers, -1) # 12 * (4 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb

class CNAPWithBNAdapterAdaptInputOutput(LoadPretrainedBaseMixIn,
                                        InitAdaptForInputOutputMixIn,
                                        GenShiftScaleParamsForInputOutputMixIn,
                                        CNAP):
    """ CNAP with BERT+BN adapter. Apply adaptation on input/output of the bn
        adapters. 
    """
    def __init__(self, args, bert_name, bert_config, pt_encoder_state_dict=None, pt_encoder_dist_metric=None):
        super().__init__(args, bert_name, bert_config)

        self.pretrained_state_dict = pt_encoder_state_dict
        self.load_pretrained_base_model()
        # Override distance metric by the one used by pretrained encoder
        self.dist_metric = dist_metric_by_name(pt_encoder_dist_metric)

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptInputOutput

    def init_dist_metric(self):
        return EuclideanDist(dim=-1)

    def init_task_emb_model(self):
        return nn.RNN(16*768*4, self.args.task_emb_size, 2, nonlinearity='tanh', batch_first=True)

    def _task_encoding(self, task):
        raise NotImplementedError()

class CNAPWithBNAdapterAdaptInputOutput_AR(GenTaskEmbForShiftScaleMixIn, 
                                           CNAPWithBNAdapterAdaptInputOutput):
    """ CNAP with BERT+BN adapter. Apply adaptation on input/output of the bn
        adapters. 
    """
    def __init__(self, args, bert_name, bert_config, pt_encoder_state_dict=None, pt_encoder_dist_metric=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptInputOutputTaskEmb

    def setup_grad(self):
        if self.args.cnap_freeze_base_model:
            # Freenze the BERT model with linear
            for n, p in self.bert.named_parameters():
                if 'shift' in n or 'scale' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            for p in self.linear.parameters():
                p.requires_grad = False
        else:
            # Only bottleneck adapters are trainable
            for n, p in self.bert.named_parameters():
                if not 'adapter' in n and not ('shift' in n or 'scale' in n):
                    p.requires_grad = False

class CNAPWithBNAdapterAdaptInputOutput_AR_FIM(CNAPWithBNAdapterAdaptInputOutput_AR):
    """ Use `batch version` of FIM calculation, should be wrong """
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_state_dict=pt_encoder_state_dict, pt_encoder_dist_metric=pt_encoder_dist_metric)

    def forward_task(self, task):
        loss, acc = super().forward_task(task)
        loss += 0.001 * self.bert.regularization_term()
        return loss, acc

    def _task_encoding(self, task, num_steps=5):
        for n, p in self.bert.named_parameters():
            if 'adapter' in n and not ('shift' in n or 'scale' in n):
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        
        num_classes = task['labels'].unique().shape[0]
        num_examples = task['labels'].shape[0]
        num_support = num_examples // 2
       
        for i in range(num_steps):
            embs = self.bert(**{k:v for k,v in task.items() if k in self.bert_arg_list})[1]
            # embs = self.dropout(embs)
            embs = self.linear(embs)

            prototypes = self.prototype_build(embs[:num_support],
                                              task['labels'][:num_support])
            assert num_classes == prototypes.shape[0]
            query_embs = embs[num_support:]
            num_query = query_embs.shape[0]
            query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                    prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

            target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
            loss = self.loss(query_logits, target)
            # loss = self.loss(query_logits, task['labels'][num_support:])
            # acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
            self.bert.zero_grad()
            loss.backward()
            for n, p in self.bert.named_parameters():
                if 'adapter' in n and not ('shift' in n or 'scale' in n):
                    assert p.grad is not None
                    p.grad_accumu += p.grad.data ** 2 
                    p.grad_accumu_count += 1
        
        for n,p in self.bert.named_parameters():
            if 'adapter' in n and not ('shift' in n or 'scale' in n):
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False
        
        all_grads = [] 
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))
        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(self.bert.config.num_hidden_layers, -1) # 12 * (4 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb

class CNAPWithBNAdapterAdaptInputOutput_AR_FIM_New(CNAPWithBNAdapterAdaptInputOutput_AR):
    """ Use correct FIM calculation.  """
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_state_dict=pt_encoder_state_dict, pt_encoder_dist_metric=pt_encoder_dist_metric)

    def forward_task(self, task):
        loss, acc = super().forward_task(task)
        loss += 0.001 * self.bert.regularization_term()
        return loss, acc

    def _task_encoding(self, task, num_steps=5):
        for n, p in self.bert.named_parameters():
            if 'adapter' in n and not ('shift' in n or 'scale' in n):
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        
        num_classes = task['labels'].unique().shape[0]
        num_examples = task['labels'].shape[0]
        num_support = num_examples // 2
       
        for _ in range(num_steps):
            embs = self.bert(**{k:v for k,v in task.items() if k in self.bert_arg_list})[1]
            # embs = self.dropout(embs)
            embs = self.linear(embs)

            prototypes = self.prototype_build(embs[:num_support],
                                              task['labels'][:num_support])
            assert num_classes == prototypes.shape[0]
            for ind in range(num_support, num_examples):
                query_embs = embs[ind:ind+1]
                num_query = 1
                query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                        prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

                target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
                loss = self.loss(query_logits, target)
                # loss = self.loss(query_logits, task['labels'][num_support:])
                # acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
                self.bert.zero_grad()
                loss.backward(retain_graph=True)
                for n, p in self.bert.named_parameters():
                    if 'adapter' in n and not ('shift' in n or 'scale' in n):
                        assert p.grad is not None
                        p.grad_accumu += p.grad.data ** 2 
                        p.grad_accumu_count += 1
        
        for n,p in self.bert.named_parameters():
            if 'adapter' in n and not ('shift' in n or 'scale' in n):
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False
        
        all_grads = [] 
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))

        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(self.bert.config.num_hidden_layers, -1) # 12 * (4 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb

#########################################################
#   CNAP with BNAdapter and context vector adaptation   #
#########################################################
class CNAPWithBNAdapterAdaptContext(LoadPretrainedBaseMixIn,
                                                # InitAdaptForInputOutputMixIn,
                                                GenTaskEmbForEachAdapterMixIn,
                                                CNAP):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config)

        if pt_encoder_state_dict is not None:
            self.pretrained_state_dict = pt_encoder_state_dict
            self.load_pretrained_base_model()
        # Override distance metric by the one used by pretrained encoder
        self.dist_metric = dist_metric_by_name(pt_encoder_dist_metric)

        self.disable_bert_dropout = False
        self.grad_num_steps = 5

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmb

    def setup_grad(self):
        if self.args.cnap_freeze_base_model:
            # Freenze the BERT model with linear
            for n, p in self.bert.named_parameters():
                if 'context' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            # Only bottleneck adapters are trainable
            for n, p in self.bert.named_parameters():
                if not 'adapter' in n and not 'context' in n:
                    p.requires_grad = False

        if self.args.cnap_freeze_linear_layer:
            # Whether to freeze the last linear layer.
            for p in self.linear.parameters():
                p.requires_grad = False
        else:
            for p in self.linear.parameters():
                p.requires_grad = True

    def init_task_emb_model(self):
        return nn.GRU(16*768*2, self.args.task_emb_size, 2, batch_first=True)

    def forward_task(self, task):
        loss, acc = super().forward_task(task)
        return loss, acc

    def _task_encoding(self, task, use_abs=True):
        if self.disable_bert_dropout:
            is_bert_training = self.bert.training
            self.bert.eval()

        for n, p in self.bert.named_parameters():
            if 'adapter' in n and not 'context' in n:
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        
        num_classes = task['labels'].unique().shape[0]
        num_examples = task['labels'].shape[0]
        num_support = num_examples // 2
       
        for i in range(self.grad_num_steps):
            # for ind in range(num_support, num_examples):
            embs = self._text_encoding_by_batch(task, None, None, 100)

            prototypes = self.prototype_build(embs[:num_support],
                                            task['labels'][:num_support])
            assert num_classes == prototypes.shape[0]
            query_embs = embs[num_support:]
            num_query = query_embs.shape[0]
            query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                    prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

            target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
            loss = self.loss(query_logits, target)
            # loss = self.loss(query_logits, task['labels'][num_support:])
            # acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
            self.bert.zero_grad()
            loss.backward()
            for n, p in self.bert.named_parameters():
                if 'adapter' in n and not 'context' in n:
                    assert p.grad is not None
                    if use_abs:
                        p.grad_accumu += p.grad.data.abs()
                    else:
                        p.grad_accumu += p.grad.data ** 2 
                    p.grad_accumu_count += 1
        
        for n,p in self.bert.named_parameters():
            if 'adapter' in n and not 'context' in n:
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False

        if self.disable_bert_dropout:
            if is_bert_training:
                self.bert.train()
            else:
                self.bert.eval()
        
        all_grads = [] 
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))

        all_grads = torch.stack(all_grads) # 48 * 12288
        return self.encode_grad(all_grads)

    def encode_grad(self, all_grads):
        all_grads = all_grads.reshape(2 * self.bert.config.num_hidden_layers, -1) # 24 * (2 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb # 24 * -1
"""
class CNAPWithBNAdapterAdaptContextCLS(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLS
    
"""
class CNAPWithBNAdapterAdaptContextCLSShiftScale(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScale

"""
class CNAPWithBNAdapterCLSAROut(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = False
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

class CNAPWithBNAdapterAdaptContextCLSShiftScaleARLNorm(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        self.disable_bert_dropout = True
        self.grad_num_steps = 1
        for n, p in self.bert.named_parameters():
            if 'LayerNorm' in n:
                p.requires_grad = True

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR
"""

class CNAPWithBNAdapterAdaptContextCLSShiftScaleAR(CNAPWithBNAdapterAdaptContext):
                            #InitAdaptForInputOutputMixIn, #TODO Add this to use previous pretrained model, delete it later!
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

class CNAPWithBNAdapter_AdaptContextShiftScaleAR(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapter_AdaptContextShiftScaleAR

class CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_X(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

    def init_task_emb_model(self):
        return nn.Sequential(
            nn.Linear(self.args.bert_linear_size, self.args.bert_linear_size // 2),
            nn.ELU(),
            nn.Linear(self.args.bert_linear_size // 2, self.args.task_emb_size))

    def _task_encoding(self, task):
        num = task['input_ids'].shape[0]
        features = self._text_encoding_by_batch( task, None, None,
                                                 batch_size=min(num, 100),
                                                 cpu=False)
        features = features.mean(dim=0, keepdim=True).detach() # 1 * -1
        return self.task_emb_model(features).expand(24, -1)

class CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_X_Grad(CNAPWithBNAdapterAdaptContextCLSShiftScaleAR):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        self.x_encoder =  nn.Sequential(
                            nn.Linear(self.args.bert_linear_size, self.args.bert_linear_size // 2),
                            nn.ELU(),
                            nn.Linear(self.args.bert_linear_size // 2, self.args.task_emb_size // 2))

    def init_task_emb_model(self):
        return nn.GRU(16*768*2, self.args.task_emb_size // 2, 2, batch_first=True)

    def _task_encoding(self, task):
        grad_emb = super()._task_encoding(task)
        num = task['input_ids'].shape[0]
        features = self._text_encoding_by_batch( task, None, None,
                                                 batch_size=min(num, 100),
                                                 cpu=False)
        features = features.mean(dim=0, keepdim=True).detach() # 1 * -1
        text_emb = self.x_encoder(features).expand(24, -1)
        return torch.cat((grad_emb, text_emb), dim=-1) # 24 x 2*100

class CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_XY(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        # for label embedding
        self._text_bert = HuggingfaceBertModel.from_pretrained(
                                bert_name,
                                from_tf=False,
                                config=bert_config,
                                cache_dir=args.cache_dir if args.cache_dir else None)
        for p in self._text_bert.parameters():
            p.requires_grad = False

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

    def init_task_emb_model(self):
        return nn.Sequential(
            nn.Linear(1024, self.args.bert_linear_size),
            nn.ELU(),
            nn.Linear(self.args.bert_linear_size, self.args.task_emb_size))
    
    def _task_encoding(self, task):
        label_features = task['label_features']
        label_emb = self._text_bert(*label_features[:3])[1].detach()

        num = task['input_ids'].shape[0]
        features = self._text_encoding_by_batch( task, None, None, 
                                                batch_size=min(num, 100),
                                                cpu=False)
        features = features.mean(dim=0, keepdim=True).detach()
        return self.task_emb_model(torch.cat((label_emb, features), dim=-1)).expand(24, -1)


class CNAPWithBNAdapterAdaptContextCLSShiftScaleAR_Shared(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = True
        bert_config.use_given_context_linears = True

        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        self.disable_bert_dropout = True
        self.grad_num_steps = 3
        for n, p in self.bert.named_parameters():
            if 'LayerNorm' in n:
                p.requires_grad = True

        num_adp = bert_config.num_hidden_layers*2
        _pos = torch.zeros((num_adp, self.args.task_emb_size))
        _pos[range(num_adp), range(num_adp)] = 1.
        self.adapter_pos_encoding = nn.Parameter(_pos, requires_grad=False) # 24 * 24

        self.init_context_layers()

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

    def init_context_layers(self):
        config = self.bert.config
        self.context_linear_mid_shift = self.mlp_init_last_with_zeros(config.task_emb_size+config.hidden_size, config.bn_adapter_hidden_size)
        self.context_linear_mid_scale = self.mlp_init_last_with_zeros(config.task_emb_size+config.hidden_size, config.bn_adapter_hidden_size)
        self.context_linear_out_shift = self.mlp_init_last_with_zeros(config.task_emb_size+config.hidden_size, config.hidden_size)
        self.context_linear_out_scale = self.mlp_init_last_with_zeros(config.task_emb_size+config.hidden_size, config.hidden_size)

    def _shift_scale_params(self, task_emb): # 24 * 24

        task_emb = task_emb + self.adapter_pos_encoding # 24 * 24

        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({ 'attention': {'task_emb': task_emb[i*2],
                                                    "context_linear_mid_shift": self.context_linear_mid_shift,
                                                    "context_linear_mid_scale": self.context_linear_mid_scale,
                                                    "context_linear_out_shift": self.context_linear_out_shift,
                                                    "context_linear_out_scale": self.context_linear_out_scale}, 
                                     'output': {'task_emb': task_emb[i*2+1],
                                                "context_linear_mid_shift": self.context_linear_mid_shift,
                                                "context_linear_mid_scale": self.context_linear_mid_scale,
                                                "context_linear_out_shift": self.context_linear_out_shift,
                                                "context_linear_out_scale": self.context_linear_out_scale}
                                    })
        return param_dict_list

    def mlp_init_last_with_zeros(self, input_size, output_size):
        last_linear = nn.Linear(input_size // 2, output_size)
        last_linear.weight.data.fill_(0.)
        last_linear.bias.data.fill_(0.)
        m_list = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ELU(),
            last_linear)
        return m_list

###################################################
#    CNAP with pretrained task embedding model    #
###################################################
class CNAPCLSShiftScaleARPretrainedTaskEmb(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = False
        bert_config.adapt_out = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        self.disable_bert_dropout = True
        self.grad_num_steps = 1
        for n, p in self.bert.named_parameters():
            if 'LayerNorm' in n:
                p.requires_grad = True

        for n, p in self.task_emb_model.named_parameters():
            p.requires_grad = False

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

    def init_task_emb_model(self):
        return nn.Linear(48 * 12288, self.args.task_emb_size)
    
    def _task_encoding(self, task):
        return super()._task_encoding(task, use_abs=False)

    def encode_grad(self, all_grads):
        all_grads = all_grads.reshape(-1).unsqueeze(0) # 1 * (48 * 12288)
        task_emb = self.task_emb_model(all_grads) # 1 * 100
        return task_emb.expand(24, self.args.task_emb_size)

class CNAPWithBNAdapterAdaptContextCLSShiftScaleARLarger(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR


##########################
#      Hypernetwork      #
##########################
class HyperWithBNAdapter(CNAPWithBNAdapterAdaptContext):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        self.disable_bert_dropout = True
        self.grad_num_steps = 3
        self.init_hyper_net()

        num_adp = bert_config.num_hidden_layers*2
        _pos = torch.zeros((num_adp, num_adp))
        _pos[range(num_adp), range(num_adp)] = 1.
        self.adapter_pos_encoding = nn.Parameter(_pos, requires_grad=False)

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapter

    def setup_grad(self):
        if self.args.cnap_freeze_base_model:
            # Freenze the BERT model with linear
            for n, p in self.bert.named_parameters():
                if not 'LayerNorm' in n:
                    p.requires_grad = False
        else:
            # Only bottleneck adapters are trainable
            for n, p in self.bert.named_parameters():
                if not 'adapter' in n and not 'context' in n:
                    p.requires_grad = False

    def init_task_emb_model(self):
        return nn.RNN(16*768*2, 24, 1, batch_first=True, nonlinearity='relu')

    def _linear_init_zero(self, in_d, out_d):
        l = nn.Linear(in_d, out_d)
        torch.nn.init.zeros_(l.weight)
        torch.nn.init.zeros_(l.bias)
        return l

    def init_hyper_net(self):
        bert_size = self.bert.config.hidden_size
        self.down_proj_weight_linear = self._linear_init_zero(24, 16*bert_size)
        self.down_proj_bias_linear = self._linear_init_zero(24, 16)
        self.up_proj_weight_linear = self._linear_init_zero(24, bert_size*16)
        self.up_proj_bias_linear = self._linear_init_zero(24, bert_size)
        self.ln_weight_linear = self._linear_init_zero(24, bert_size)
        self.ln_bias_linear = self._linear_init_zero(24, bert_size)

    def encode_grad(self, all_grads):
        all_grads = all_grads.reshape(2 * self.bert.config.num_hidden_layers, -1) # 24 * (2 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        output, hidden = self.task_emb_model(all_grads)
        task_emb = hidden.reshape(1, -1)
        return task_emb

    def _shift_scale_params(self, task_emb): # 1 * 24
        bert_size = self.bert.config.hidden_size
        task_emb = task_emb.expand(24, -1)
        task_emb = task_emb + self.adapter_pos_encoding # 24 * 24
        down_weight = self.down_proj_weight_linear(task_emb) # 24 * 16*bert_size
        down_weight = down_weight.reshape(24, 16, bert_size)
        down_bias = self.down_proj_bias_linear(task_emb)
        up_weight = self.up_proj_weight_linear(task_emb)
        up_weight = up_weight.reshape(24, bert_size, 16)
        up_bias = self.up_proj_bias_linear(task_emb)
        ln_weight = self.ln_weight_linear(task_emb)
        ln_bias = self.ln_bias_linear(task_emb)

        param_dict_list = []
        for i in range(self.bert.config.num_hidden_layers):
            param_dict_list.append({ 'attention': {'use_hyper_net': True,
                                                   'down_proj_weight': down_weight[i*2],
                                                   'down_proj_bias': down_bias[i*2],
                                                   'up_proj_weight': up_weight[i*2],
                                                   'up_proj_bias': up_bias[i*2],
                                                   'layer_norm_weight': ln_weight[i*2],
                                                   'layer_norm_bias': ln_bias[i*2] }, 
                                     'output': {'use_hyper_net': True,
                                                   'down_proj_weight': down_weight[i*2+1],
                                                   'down_proj_bias': down_bias[i*2+1],
                                                   'up_proj_weight': up_weight[i*2+1],
                                                   'up_proj_bias': up_bias[i*2+1],
                                                   'layer_norm_weight': ln_weight[i*2+1],
                                                   'layer_norm_bias': ln_bias[i*2+1] }
                                     })
        return param_dict_list

class HyperWithBNAdapterPretrainedTaskEmb(HyperWithBNAdapter):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
        self.grad_num_steps = 1
        for p in self.task_emb_model.parameters():
            p.requires_grad = False
    
    def init_task_emb_model(self):
        return nn.Linear(48 * 12288, self.args.task_emb_size)
    
    def _task_encoding(self, task):
        return super()._task_encoding(task, use_abs=False)

    def encode_grad(self, all_grads):
        all_grads = all_grads.reshape(-1).unsqueeze(0) # 1 * (48 * 12288)
        return self.task_emb_model(all_grads) # 1 * 100

# **************************************************************************** #

class CNAPWithBNAdapterAdaptContextMLPTaskEmb(LoadPretrainedBaseMixIn,
                                                InitAdaptForInputOutputMixIn,
                                                GenTaskEmbForEachAdapterMixIn,
                                                CNAP):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        super().__init__(args, bert_name, bert_config)

        self.pretrained_state_dict = pt_encoder_state_dict
        self.load_pretrained_base_model()
        # Override distance metric by the one used by pretrained encoder
        self.dist_metric = dist_metric_by_name(pt_encoder_dist_metric)

        self.grad_linear = nn.Linear(16*768*2, 100)

    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmb

    def setup_grad(self):
        if self.args.cnap_freeze_base_model:
            # Freenze the BERT model with linear
            for n, p in self.bert.named_parameters():
                if 'context' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            for p in self.linear.parameters():
                p.requires_grad = False
        else:
            # Only bottleneck adapters are trainable
            for n, p in self.bert.named_parameters():
                if not 'adapter' in n and not 'context' in n:
                    p.requires_grad = False

    def init_task_emb_model(self):
        return nn.GRU(100, self.args.task_emb_size, 2, batch_first=True)

    def forward_task(self, task):
        loss, acc = super().forward_task(task)
        return loss, acc

    def _task_encoding(self, task, num_steps=5):
        is_bert_training = self.bert.training
        self.bert.eval()
        for n, p in self.bert.named_parameters():
            if 'adapter' in n and not 'context' in n:
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        
        num_classes = task['labels'].unique().shape[0]
        num_examples = task['labels'].shape[0]
        num_support = num_examples // 2
       
        for i in range(num_steps):
            # for ind in range(num_support, num_examples):
            embs = self._text_encoding_by_batch(task, None, None, 100)

            prototypes = self.prototype_build(embs[:num_support],
                                            task['labels'][:num_support])
            assert num_classes == prototypes.shape[0]
            query_embs = embs[num_support:]
            num_query = query_embs.shape[0]
            query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                    prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

            target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
            loss = self.loss(query_logits, target)
            # loss = self.loss(query_logits, task['labels'][num_support:])
            # acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
            self.bert.zero_grad()
            loss.backward()
            for n, p in self.bert.named_parameters():
                if 'adapter' in n and not 'context' in n:
                    assert p.grad is not None
                    p.grad_accumu += p.grad.data ** 2 
                    # p.grad_accumu += p.grad.data.abs()
                    p.grad_accumu_count += 1
        
        for n,p in self.bert.named_parameters():
            if 'adapter' in n and not 'context' in n:
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False
        if is_bert_training:
            self.bert.train()
        else:
            self.bert.eval()
        
        all_grads = [] 
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))

        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(2 * self.bert.config.num_hidden_layers, -1) # 24 * (2 * 12288)
        all_grads = self.grad_linear(all_grads) # 24 * 100
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb

class CNAPWithBNAdapterAdaptContextCLSMLPTaskEmbShiftScaleAR(CNAPWithBNAdapterAdaptContextMLPTaskEmb):
    def __init__(self, args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict=None):
        bert_config.multilayer_hyper = True
        super().__init__(args, bert_name, bert_config, pt_encoder_dist_metric, pt_encoder_state_dict)
    def set_bert_type(self):
        self.bert_class = BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR

class CNAPWithBNAdapterV0_1(CNAPWithBNAdapter):
    """ CNAP with Fisher Information Matrix (FIM) as task representation. FIM is
        calculated after trained a linear layer given a task.
    
        FIM is calculated with BERT plus linear classification layer. BERT and the
        linear layer are first fine tuned on the training data of a task. After 
        reaching to more than 90% accuracy or a specific number of epochs, we 
        calculate the expected squared gradients of the BERT parameters as the 
        diagnal values of the FIM and use them as task representation.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict=pt_bert_bn_state_dict)
        self.task_emb_bert = BertModelWithBNAdapter.from_pretrained(
                                                    bert_name,
                                                    from_tf=False,
                                                    config=bert_config,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
        self.task_emb_linear = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)

        for p1, p2 in zip(self.task_emb_bert.parameters(), self.bert.parameters()):
            p1.data.copy_(p2.data)
            p1.requires_grad = False
        for p1, p2 in zip(self.task_emb_linear.parameters(), self.linear.parameters()):
            p1.data.copy_(p2.data)
            p1.requires_grad = False

    def gen_adapt_param(self, task, num_support, batch_size, detach=False):
        if self.args.cnap_adapt:
            task_emb = self._task_encoding({
                k:v[:num_support] if type(v) is torch.Tensor and len(v.shape) > 0 else v for k,v in task.items()})
            if detach:
                task_emb = task_emb.detach()
            adapt_param_list = self._shift_scale_params(task_emb)
            return ['params_dict_list'], [adapt_param_list]
        else:
            return None, None

    def _task_encoding(self, task, num_steps=5, num_epochs=30):
        task_classify_linear = nn.Linear(self.task_emb_bert.config.hidden_size,
                                              task['labels'].unique().shape[0])
        task_classify_linear.to(self.get_device())
        optimizer = AdamW(task_classify_linear.parameters(), lr=0.0002)
        
        num_examples = task['input_ids'].shape[0]

        batch_size = min(20, num_examples)
        for e in range(num_epochs):
            acc_list = []
            loss_list = []
            for ind in range(0, num_examples, batch_size):
                embs = self.task_emb_bert(**{k:v[ind:ind+batch_size] for k,v in task.items() if k in self.bert_arg_list})[1]
                # embs = self.dropout(embs)
                embs = self.task_emb_linear(embs)
                logits = task_classify_linear(embs)
                loss = self.loss(logits, task['labels'][ind:ind+batch_size])
                acc = self.accuracy_fn(logits, task['labels'][ind:ind+batch_size])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc_list.append(acc.item())
                loss_list.append(loss.item())
            if np.mean(acc_list) > 0.9:
                break
        logger.debug(f'Training for task emb, total epoch {e}, acc {np.mean(acc_list)}')

        for n, p in self.task_emb_bert.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        batch_size = 1
        for ind in range(0, num_examples, batch_size):
            embs = self.task_emb_bert(**{k:v[ind:ind+batch_size] for k,v in task.items() if k in self.bert_arg_list})[1]
            embs = self.task_emb_linear(embs)
            logits = task_classify_linear(embs)
            for i in range(num_steps):
                target = torch.multinomial(F.softmax(logits, dim=-1), 1).detach().view(-1)
                loss = self.loss(logits, target)
                self.task_emb_bert.zero_grad()
                loss.backward(retain_graph=True)
                for n, p in self.task_emb_bert.named_parameters():
                    if 'adapter' in n:
                        assert p.grad is not None
                        p.grad_accumu += p.grad.data ** 2 
                        p.grad_accumu_count += 1
        for n,p in self.task_emb_bert.named_parameters():
            if 'adapter' in n:
                assert num_examples*num_steps ==  p.grad_accumu_count
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False
        
        all_grads = [] 
        for n, m in self.task_emb_bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))
        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(self.bert.config.num_hidden_layers, -1) # 12 * (4 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb

class CNAPWithBNAdapterV0_3(CNAPWithBNAdapterAdaptInputOutput):
    """ Use fisher information matrix as taks embedding.
        We don't fine tune the text encoding network before calculating FIM.
        The hypternetwork is a RNN model.
        Both the input and output to the bottleneck adapters are adapted.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict=pt_bert_bn_state_dict)

    def _task_encoding(self, task, num_steps=5):
        for n, p in self.bert.named_parameters():
            if 'adapter' in n:
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0
        
        num_classes = task['labels'].unique().shape[0]
        num_examples = task['labels'].shape[0]
        num_support = num_examples // 2
       
        for i in range(num_steps):
            embs = self.bert(**{k:v for k,v in task.items() if k in self.bert_arg_list})[1]
            # embs = self.dropout(embs)
            embs = self.linear(embs)

            prototypes = self.prototype_build(embs[:num_support],
                                              task['labels'][:num_support])
            assert num_classes == prototypes.shape[0]
            query_embs = embs[num_support:]
            num_query = query_embs.shape[0]
            query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                    prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

            target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
            loss = self.loss(query_logits, target)
            # loss = self.loss(query_logits, task['labels'][num_support:])
            # acc = self.accuracy_fn(query_logits, task['labels'][num_support:])
            self.bert.zero_grad()
            loss.backward()
            for n, p in self.bert.named_parameters():
                if 'adapter' in n:
                    assert p.grad is not None
                    p.grad_accumu += p.grad.data ** 2 
                    p.grad_accumu_count += 1
        
        for n,p in self.bert.named_parameters():
            if 'adapter' in n:
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False
        
        all_grads = [] 
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))
        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(self.bert.config.num_hidden_layers, -1) # 12 * (4 * 12288)
        all_grads = all_grads.unsqueeze(0)
        self.task_emb_model.flatten_parameters()
        task_emb = self.task_emb_model(all_grads)[0][0]
        return task_emb

class CNAPWithBNAdapterV0_4(CNAPWithBNAdapterV0_3):
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict)

    def init_dist_metric(self):
        return EuclideanDist(dim=-1)


class CNAPWithBNAdapter_TaskEmbY(CNAPWithBNAdapter):
    """ Using label representation as task embedding.
        The hypernework is simply MLP networks.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict=pt_bert_bn_state_dict)
        self._text_bert = HuggingfaceBertModel.from_pretrained(
                                bert_name,
                                from_tf=False,
                                config=bert_config,
                                cache_dir=args.cache_dir if args.cache_dir else None)
        for p in self._text_bert.parameters():
            p.requires_grad = False

    def init_task_emb_model(self):
        return nn.Sequential(
            nn.Linear(self.args.bert_linear_size, self.args.bert_linear_size // 2),
            nn.ELU(),
            nn.Linear(self.args.bert_linear_size // 2, self.args.task_emb_size))
    
    def _task_encoding(self, task):
        label_features = task['label_features']
        label_emb = self._text_bert(*label_features[:3])[1]
        return self.task_emb_model(label_emb)


class CNAPWithBNAdapterAdaptInputOutput_TaskEmbX(CNAPWithBNAdapterAdaptInputOutput):
    """ Using average input representation as task embedding.
        The hypernework is simply MLP networks.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict=pt_bert_bn_state_dict)
    
    def init_task_emb_model(self):
        return nn.Sequential(
            nn.Linear(self.args.bert_linear_size, self.args.bert_linear_size // 2),
            nn.ELU(),
            nn.Linear(self.args.bert_linear_size // 2, self.args.task_emb_size))
    
    def _task_encoding(self, task):
        num = task['input_ids'].shape[0]
        features = self._text_encoding_by_batch(
            task,
            None,
            None,
            min(num, 100),
            cpu=False
        )
        features = features.mean(dim=0, keepdim=True).detach()
        return self.task_emb_model(features)

class CNAPWithBNAdapterAdaptInputOutput_TaskEmbY(CNAPWithBNAdapterAdaptInputOutput):
    """ Using label representation as task embedding.
        The hypernework is simply MLP networks.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict=pt_bert_bn_state_dict)
        self._text_bert = HuggingfaceBertModel.from_pretrained(
                                bert_name,
                                from_tf=False,
                                config=bert_config,
                                cache_dir=args.cache_dir if args.cache_dir else None)
        for p in self._text_bert.parameters():
            p.requires_grad = False

    def init_task_emb_model(self):
        return nn.Sequential(
            nn.Linear(self.args.bert_linear_size, self.args.bert_linear_size // 2),
            nn.ELU(),
            nn.Linear(self.args.bert_linear_size // 2, self.args.task_emb_size))
    
    def _task_encoding(self, task):
        label_features = task['label_features']
        label_emb = self._text_bert(*label_features[:3])[1]
        return self.task_emb_model(label_emb)

class CNAPWithBNAdapterAdaptInputOutput_TaskEmbXY(CNAPWithBNAdapterAdaptInputOutput):
    """ Using the concatenation of average input representation and the label 
        representation as task embedding. The hypernework is simply MLP networks.
    """
    def __init__(self, args, bert_name, bert_config, pt_bert_bn_state_dict):
        super().__init__(args, bert_name, bert_config, pt_bert_bn_state_dict=pt_bert_bn_state_dict)
        self._text_bert = HuggingfaceBertModel.from_pretrained(
                                bert_name,
                                from_tf=False,
                                config=bert_config,
                                cache_dir=args.cache_dir if args.cache_dir else None)
        for p in self._text_bert.parameters():
            p.requires_grad = False

    def init_task_emb_model(self):
        return nn.Sequential(
            nn.Linear(self.args.bert_linear_size * 2, self.args.bert_linear_size // 2),
            nn.ELU(),
            nn.Linear(self.args.bert_linear_size // 2, self.args.task_emb_size))
    
    def _task_encoding(self, task):
        label_features = task['label_features']
        label_emb = self._text_bert(*label_features[:3])[1].detach()

        num = task['input_ids'].shape[0]
        features = self._text_encoding_by_batch(task, None, None,
                                                batch_size=min(num, 100), cpu=False)
        features = features.mean(dim=0, keepdim=True).detach()
        return self.task_emb_model(torch.cat((label_emb, features), dim=-1))
