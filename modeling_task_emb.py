from __future__ import absolute_import, division, print_function

from logging import raiseExceptions
from typing import ValuesView
from modeling_cnap import CNAP, GenTaskEmbForEachAdapterMixIn, InitAdaptForInputOutputMixIn, LoadPretrainedBaseMixIn
from torch import autograd
from utils import MyDataParallel, dist_metric_by_name
from sklearn.metrics import roc_auc_score
from modeling_adaptation import PrototypeBuildingNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling_bert_adaptation import BertModelWithBNAdapter, BertModelWithBNAdapterAdaptContextTaskEmb
from modeling_mixin import ClassificationAccMixin, CrossEntropyMixin, GetDeviceMixin
from transformers.modeling_bert import BertModel as HuggingfaceBertModel

from logging_utils import get_logger
logger = get_logger('TaskEmb Model')


class TaskembModel(CrossEntropyMixin, ClassificationAccMixin, GetDeviceMixin, nn.Module):
    """ CNAP with bottleneck adapters for diverse text classification tasks. """

    bert_arg_list = ['input_ids', 'attention_mask', 'token_type_ids']
    task_arg_list = ['labels'] + bert_arg_list

    def __init__(self, args, bert_name, bert_config):
        super().__init__()
        self.args = args
        self.init_bert(bert_name, bert_config)
        self.prototype_build = PrototypeBuildingNetwork()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.init_task_emb_model()

        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def init_bert(self):
        raise NotImplementedError()

    def init_task_emb_model(self):
        raise NotImplementedError()

    def _task_encoding(self, task):
        raise NotImplementedError()
    
    def set_bert_dataparallel(self):
        self.bert = MyDataParallel(self.bert)

    def load_pretrained_encoder(self, pretrained_state_dict, dist_metric):
        self.dist_metric = dist_metric_by_name(dist_metric)
        state_dict = self.state_dict()
        for pt_n, pt_p in pretrained_state_dict.items():
            if pt_n.split('.')[0] == 'bert' or pt_n.split('.')[0] == 'linear':
                state_dict[pt_n].copy_(pt_p.data)

    def forward(self, batch, eval=False):
        device = self.get_device()
        device_name = self.get_device_name()
        if hasattr(batch, device_name):
            local_batch = getattr(batch, device_name)

            task_emb_dict = {}
            for i,task_list in enumerate(local_batch): 
                task_emb_dict[i] = [self.forward_task(task).squeeze(0) for task in task_list] # num_task * task_emb_dim

            anchors = []
            pos_neg_items = []
            for task_id in task_emb_dict:
                for anchor_index in range(len(task_emb_dict[task_id]) - 1):
                    for _index in range(anchor_index+1, len(task_emb_dict[task_id])):
                        # anchor item
                        anchors.append(task_emb_dict[task_id][anchor_index])
                        _pn_items = [] 
                        # positive item
                        _pn_items.append(task_emb_dict[task_id][_index])

                        # negative items
                        for neg_task_id in task_emb_dict:
                            if neg_task_id != task_id:
                                _pn_items.append(task_emb_dict[neg_task_id][_index])
                        pos_neg_items.append(torch.stack(_pn_items))
            anchors = torch.stack(anchors) # num_anchor * task_emb_dim
            pos_neg_items = torch.stack(pos_neg_items) # num_anchor * num_pair_per_anchor * task_emb_dim
            num_pair_per_anchor = pos_neg_items.shape[1]
            anchors = anchors.unsqueeze(1).expand(-1, num_pair_per_anchor, -1) # num_anchor * num_pair_per_anchor * task_emb_dim
            cos_sim = self.cos_sim(anchors, pos_neg_items) # num_anchor * num_pair_per_anchor
            labels = torch.zeros(cos_sim.shape[0], dtype=torch.long, device=device)

            if not eval:
                loss = self.loss(cos_sim, labels)
                pred = cos_sim.detach().cpu()
                labels = labels.cpu()
                acc = torch.sum(pred.argmax(dim=-1) == labels).float() / labels.shape[0]
                logger.info(f"Acc: {acc*100:.2f}%")
                # auc = roc_auc_score(labels, pred)
                acc = torch.tensor([acc]).to(device)
                return loss.unsqueeze(0), acc
            else:
                return cos_sim.detach(), labels
        else:
            assert eval # this could happen only during evaluation with multiple gpu
            loss, query_acc = torch.tensor(0.).to(device), torch.tensor([-1.]).to(device)
            return loss, query_acc
    
    def extract_gradients(self, batch):
        device_name = self.get_device_name()

        if hasattr(batch, device_name):
            local_batch = getattr(batch, device_name)

            task_grad_list = []
            for i, task in enumerate(local_batch):
                task_grad_1, task_grad_2 = self.cal_task_grad(task)
                task_grad_list.append(torch.stack((task_grad_1, task_grad_2)))
            
            return torch.stack(task_grad_list)
        else:
            raise NotImplementedError()

    def extract_task_embs(self, batch):
        device_name = self.get_device_name()

        if hasattr(batch, device_name):
            local_batch = getattr(batch, device_name)

            task_emb_list = []
            for i, task in enumerate(local_batch):
                task_emb= self.forward_task(task)
                task_emb_list.append(task_emb)
            import ipdb; ipdb.set_trace() 
            return torch.stack(task_emb_list)
        else:
            raise NotImplementedError()

    def forward_task_deprecated(self, task):
        task_labels = task['labels']
        if not ('num_classes' in task and 'num_shots' in task):
            num_classes = task_labels.unique().shape[0]
            num_shots = self.args.num_shots_support
        else:
            num_classes = task['num_classes'].item()
            num_shots = task['num_shots'].item()
        num_support = num_classes * num_shots
        num_query = task_labels.shape[0] - num_support

        return self._task_encoding({k:v[:num_support] for k,v in task.items() if k in self.task_arg_list}), \
               self._task_encoding({k:v[num_support:] for k,v in task.items() if k in self.task_arg_list}) # 1 * 100

    def forward_task(self, task):
        return self._task_encoding({k:v for k,v in task.items() if k in self.task_arg_list})

    def cal_task_grad(self, task):
        task_labels = task['labels']
        if not ('num_classes' in task and 'num_shots' in task):
            num_classes = task_labels.unique().shape[0]
            num_shots = self.args.num_shots_support
        else:
            num_classes = task['num_classes'].item()
            num_shots = task['num_shots'].item()
        num_support = num_classes * num_shots
        num_query = task_labels.shape[0] - num_support

        return self._task_grad({k:v[:num_support] for k,v in task.items() if k in self.task_arg_list}), \
               self._task_grad({k:v[num_support:] for k,v in task.items() if k in self.task_arg_list}) # 24 * -1

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
        # embs = self.dropout(embs)
        embs = self.linear(embs)
        return embs

class FIMWithBNAdapter(TaskembModel):
    def __init__(self, args, bert_name, bert_config, grad_num_steps=5):
        super().__init__(args, bert_name, bert_config)
        self.grad_num_steps = grad_num_steps

    def init_bert(self, bert_name, bert_config):
        self.bert = BertModelWithBNAdapter.from_pretrained(
                        bert_name,
                        from_tf=False,
                        config=bert_config,
                        cache_dir=self.args.cache_dir if self.args.cache_dir else None)
        self.linear = nn.Linear(bert_config.hidden_size, self.args.bert_linear_size)
        self.dist_metric = dist_metric_by_name('euc')

        for n, p in self.bert.named_parameters():
            p.requires_grad = False
        for n, p in self.linear.named_parameters():
            p.requires_grad = False
        
    def init_task_emb_model(self):
        self.task_emb_model = nn.Linear(48 * 12288, self.args.task_emb_size)

    def _task_grad(self, task):
        fim_param_check = lambda n: 'adapter' in n
        for n, p in self.bert.named_parameters():
            if fim_param_check(n):
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0

        num_classes = task['labels'].unique().shape[0]
        num_examples = task['labels'].shape[0]
        num_support = num_examples // 2

        for i in range(self.grad_num_steps):
            for ind in range(num_support, num_examples):
                embs = self._text_encoding_by_batch(task, None, None, 100)

                prototypes = self.prototype_build(embs[:num_support],
                                                task['labels'][:num_support])
                assert num_classes == prototypes.shape[0]
                query_embs = embs[ind:ind+1]
                num_query = 1
                query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                        prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

                target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)
                loss = self.loss(query_logits, target)

                self.bert.zero_grad()
                loss.backward()
                # grads = autograd.grad(loss, [p for n, p in self.bert.named_parameters() if 'adapter' in n])
                for n, p in self.bert.named_parameters():
                    if fim_param_check(n):
                        if p.grad is None:
                            raise ValueError(f'gradient of {n} is none')
                        p.grad_accumu += p.grad.data ** 2
                        p.grad_accumu_count += 1

        for n,p in self.bert.named_parameters():
            if fim_param_check(n):
                p.requires_grad = False
                p.grad_accumu /= p.grad_accumu_count

        all_grads = []
        for n, m in self.bert.named_modules():
            if hasattr(m, 'weight') and hasattr(m.weight, 'grad_accumu'):
                grad = m.weight.grad_accumu
                all_grads.append(grad.reshape(-1))

        all_grads = torch.stack(all_grads) # 48 * 12288
        all_grads = all_grads.reshape(-1) # 24 * (2 * 12288)

        return all_grads

    def _task_encoding(self, task, num_steps=5):
        grads = self._task_grad(task, num_steps)
        task_emb = self.task_emb_model(grads.unsqueeze(0))
        return task_emb

class FIMWithBNAdapterBatch(FIMWithBNAdapter):
    def __init__(self, args, bert_name, bert_config, grad_num_steps=5):
        super().__init__(args, bert_name, bert_config)
        self.use_true_labels = False
        self.grad_num_steps = grad_num_steps

    def _task_grad(self, batch):
        is_bert_training = self.bert.training
        self.bert.eval()
        fim_param_check = lambda n: 'adapter' in n
        for n, p in self.bert.named_parameters():
            if fim_param_check(n):
                p.requires_grad = True
                p.grad_accumu = torch.zeros_like(p.data)
                p.grad_accumu_count = 0

        num_classes = batch['labels'].unique().shape[0]
        num_examples = batch['labels'].shape[0]
        num_shots = num_examples // num_classes
        batch_split = []
        if num_shots > 10:
            split_num = num_examples // 2
            for i in range(0, num_examples, split_num):
                batch_split.append({k:v[i:i+split_num] for k,v in batch.items()})
        else:
            batch_split.append(batch)

        for i in range(self.grad_num_steps):
            for task in batch_split:
                num_examples = task['labels'].shape[0]
                num_support = num_examples // 2
                embs = self._text_encoding_by_batch(task, None, None, 50)

                prototypes = self.prototype_build(embs[:num_support],
                                                task['labels'][:num_support])
                assert num_classes == prototypes.shape[0]
                query_embs = embs[num_support:]
                num_query = query_embs.shape[0]
                query_logits = self.dist_metric(query_embs.unsqueeze(1).expand(-1, num_classes, -1), # num_query * num_classes * hidden_size
                                        prototypes.expand(num_query, -1, -1)) # num_query * num_classes * hidden_size

                if self.use_true_labels:
                    target = task['labels'][num_support:]
                else:
                    target = torch.multinomial(F.softmax(query_logits, dim=-1), 1).detach().view(-1)

                loss = self.loss(query_logits, target)

                self.bert.zero_grad()
                loss.backward()
                # grads = autograd.grad(loss, [p for n, p in self.bert.named_parameters() if 'adapter' in n])
                for n, p in self.bert.named_parameters():
                    if fim_param_check(n):
                        if p.grad is None:
                            raise ValueError(f'gradient of {n} is none')
                        p.grad_accumu += p.grad.data ** 2
                        p.grad_accumu_count += 1

        for n,p in self.bert.named_parameters():
            if fim_param_check(n):
                p.grad_accumu /= p.grad_accumu_count
                p.requires_grad = False # set it back

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
        all_grads = all_grads.reshape(-1) # 24 * (2 * 12288)

        return all_grads

    def _task_encoding(self, batch):
        grads = self._task_grad(batch)
        task_emb = self.task_emb_model(grads.unsqueeze(0))
        return task_emb

class FIMWithBNAdapterBatchTrueLabels(FIMWithBNAdapterBatch):
    def __init__(self, args, bert_name, bert_config):
        super().__init__(args, bert_name, bert_config)
        self.use_true_labels = True