##################################################
# BERT model with context vectors for adaptation #
##################################################

from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from itertools import chain
from modeling_mixin import ClassificationAccMixin, CrossEntropyMixin, GetDeviceNameMixin
import sys
from transformers.optimization import AdamW
from logging_utils import get_logger
import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from wandb.sdk_py27.lib.telemetry import context

from transformers import (BertEncoder, BertModel,
                            BertForMaskedLM,
                            BertForSequenceClassification,
                            BertForQuestionAnswering,
                            BertLayer,
                            BertAttention, BertOutput, BertPreTrainedModel,
                            BertSelfAttention,
                            BertSelfOutput,
                            ACT2FN
                            )
from utils import MyDataParallel, aggregate_accuracy, linear_classifier, loss, mean_pooling
from modeling_adaptation import ( LinearClassifierAdaptationNetwork)

logger = get_logger('Leopard')

class BertSelfOutputWithContextAdapter(BertSelfOutput):
    """ Fully connected layer after self-attention layer,
        with the bottleneck adapter layer.
    """
    def __init__(self, config):
        super().__init__(config)
        # self.adapter = BottleneckAdapter(config)
    
        self.context_proj_layer = nn.Linear(config.bn_context_size,
                                         config.hidden_size)
    def forward(self, hidden_states, input_tensor, context):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Apply the adaptation layer.
        # hidden_states = self.adapter(hidden_states, context)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if context is not None:
            # context_hidden = self.context_proj_layer(context)
            context_hidden = context
            context_hidden = context_hidden[None, None, :]
            hidden_states = hidden_states + context_hidden

        return hidden_states
        
class BertOutputWithContextAdapter(BertOutput):
    """ Fully connected layer before output of transformer layer, 
        with the bottleneck adapter layer.
    """
    def __init__(self, config):
        super().__init__(config)
        # self.adapter = BottleneckAdapter(config)
        self.context_proj_layer = nn.Linear(config.bn_context_size,
                                         config.hidden_size)

    def forward(self, hidden_states, input_tensor, context=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Apply the adaptation layer.
        # hidden_states = self.adapter(hidden_states, context)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        if context is not None:
            # context_hidden = self.context_proj_layer(context)
            context_hidden = context
            context_hidden = context_hidden[None, None, :]
            hidden_states = hidden_states + context_hidden
        return hidden_states

class BertAttentionWithContextAdapter(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithContextAdapter(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                context=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states, context)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertLayerWithContextAdapter(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithContextAdapter(config)
        self.output = BertOutputWithContextAdapter(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                context=None):
        if context is None:
            self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        else:
            self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask, context=context['self'])

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        if context is None:
            layer_output = self.output(intermediate_output, attention_output)
        else:
            layer_output = self.output(intermediate_output, attention_output, context=context['out'])
        outputs = (layer_output,) + outputs
        return outputs

class BertEncoderWithContextAdapter(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithContextAdapter(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, context_list=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if context_list is None:
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
                                            context_list[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertModelWithContextAdapter(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithContextAdapter(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                context_list=None, **kwargs):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       context_list=context_list)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class BertWithContextAdapterForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertWithContextAdapterForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert_config = config

        self.bert = BertModelWithContextAdapter(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # Freeze the parameters of the text encoder (BERT)
        for param in self.bert.parameters():
            param.requires_grad = False
        # for param in self.bert.encoder.layer[-1].parameters():
        #     param.requires_grad = True
        # for layer in self.bert.encoder.layer[:-1]:
        #     layer.eval()
            
        self.bert.eval()

        self.init_weights()
        self.init_context_weights()

    def init_context_weights(self):
        for i in range(self.bert_config.num_hidden_layers):
            self_context = nn.Parameter(
                    torch.nn.init.normal_(torch.empty(self.bert_config.bn_context_size), 0, 0.001),
                    requires_grad=True
            )
            self.register_parameter(f'self_context_{i}', self_context)

            out_context = nn.Parameter(
                    torch.nn.init.normal_(torch.empty(self.bert_config.bn_context_size), 0, 0.001),
                    requires_grad=True
            )
            self.register_parameter(f'out_context_{i}', out_context)

    @property
    def context_list(self):
        context_list = []
        for i in range(self.bert_config.num_hidden_layers):
            context = {
                'self': getattr(self, f'self_context_{i}'),
                'out': getattr(self, f'out_context_{i}')
            }
            context_list.append(context)
        return context_list
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            context_list=self.context_list)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class ContextAdaptationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        input = torch.zeros(1, input_dim, input_dim)
        input[:, range(input_dim), range(input_dim)] = 1
        self.input = input
        self.num_layers = num_layers
        self.output_linear = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, task_emb):
        task_emb = task_emb[None, None, :].expand(self.num_layers, 1, task_emb.shape[-1])
        input = self.input.to(task_emb.device)
        output, _ = self.rnn(input, task_emb.contiguous())
        output = output.squeeze(0)
        output = self.output_linear(output)
        context_list = []
        for i in range(0, output.shape[0], 2):
            context_list.append({
                'self': output[i],
                'out': output[i+1]
            })
        return context_list
        
class BertWithContextAdapterForMetatraining(nn.Module, CrossEntropyMixin,
                                            ClassificationAccMixin,
                                            GetDeviceNameMixin):
    """ BERT with context vector adapters for meta-training on various tasks. """ 
    def __init__(self, args, bert_config):
        super().__init__()
        self.args = args
        self.encoder = BertModelWithContextAdapter.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool('.ckpt' in args.model_name_or_path),
                        config=bert_config,
                        cache_dir=args.cache_dir if args.cache_dir else None)
        self.adaptation_network = ContextAdaptationNetwork(
                                    input_dim=bert_config.num_hidden_layers * 2,
                                    hidden_dim=bert_config.hidden_size,
                                    output_dim=args.bn_context_size)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # Freeze the parameters of the text encoder (BERT)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
        self.classifier = linear_classifier
        # Adapation network for the linear classifier
        self.cl_adaptation_network = LinearClassifierAdaptationNetwork(bert_config.hidden_size)
        self.cl_linear = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.total_iterations = args.num_training_iterations

        self.n_gpu = 1

        # testing 
        self.optimizer = AdamW(self.parameters(), 0.0001)

    def setup_for_multiple_gpus(self, n_gpu):
        self.encoder = MyDataParallel(self.encoder)
        self.adaptation_network = self.adaptation_network
        self.cl_adaptation_network = self.cl_adaptation_network
        self.cl_linear = MyDataParallel(self.cl_linear)
        self.n_gpu = n_gpu
    
    def parameters(self):
        for model in [self.adaptation_network, self.cl_adaptation_network, self.cl_linear]:
            for p in model.parameters():
                yield p

    def named_parameters(self):
        for model in [self.adaptation_network, self.cl_adaptation_network, self.cl_linear]:
            for n, p in model.named_parameters():
                yield n, p

    def dummy_forward(self, examples):
        task_representation = self._task_encoding(examples, examples['labels'])
        context_list = self.adaptation_network(task_representation)
        # Text encoding
        if self.n_gpu > 1:
            context_list = [{
                'self': con['self'].repeat(self.n_gpu, 1),
                'out': con['out'].repeat(self.n_gpu, 1),
                } for con in context_list]
        support_features = self._text_encoding(examples, context_list)
        query_features = self._text_encoding(examples, context_list)
        # Build task-dependent linear classifier
        class_representations = self._build_class_reps(support_features, examples['labels'])
        classifier_params = self._get_classifier_params(class_representations)
        # Classify
        query_features = self.cl_linear(query_features)
        query_logits = self.classifier(query_features, classifier_params)
        loss = self.loss(query_logits, examples['labels'])
        return loss
        
    def forward(self, support_examples, query_examples):
        while True:
            query_examples = support_examples
            # Extract task representaion
            task_representation = self._task_encoding(support_examples, support_examples['labels'])
            context_list = self.adaptation_network(task_representation)
            # Text encoding
            if self.n_gpu > 1:
                context_list = [{
                    'self': con['self'].repeat(self.n_gpu, 1),
                    'out': con['out'].repeat(self.n_gpu, 1),
                    } for con in context_list]
            support_features = self._text_encoding(support_examples, context_list)
            query_features = self._text_encoding(query_examples, context_list)
            # Build task-dependent linear classifier
            class_representations = self._build_class_reps(support_features, support_examples['labels'])
            classifier_params = self._get_classifier_params(class_representations)
            # Classify
            query_features = self.cl_linear(query_features)
            query_logits = self.classifier(query_features, classifier_params)
            loss = self.loss(query_logits, query_examples['labels'])
            query_acc = self.accuracy_fn(query_logits, query_examples['labels'])

            self.adaptation_network.zero_grad()
            self.cl_adaptation_network.zero_grad()
            self.cl_linear.zero_grad()
            loss.backward()
            
            grads = [param.grad for param in chain(self.adaptation_network.parameters(),
                                                self.cl_adaptation_network.parameters(),
                                                self.cl_linear.parameters())]
            self.optimizer.step()
            print(loss, query_acc)
        return loss, query_acc, grads

    def eval(self, support_examples, query_examples):
        with torch.no_grad():
            # Extract task representaion
            task_representation = self._task_encoding(support_examples, support_examples['labels'])
            context_list = self.adaptation_network(task_representation)
            # Text encoding
            if self.n_gpu > 1:
                context_list = [{
                    'self': con['self'].repeat(self.n_gpu, 1),
                    'out': con['out'].repeat(self.n_gpu, 1),
                    } for con in context_list]
            support_features = self._text_encoding(support_examples, context_list)
            query_features = self._text_encoding(query_examples, context_list)
            # Build task-dependent linear classifier
            class_representations = self._build_class_reps(support_features, support_examples['labels'])
            classifier_params = self._get_classifier_params(class_representations)
            # Classify
            query_features = self.cl_linear(query_features)
            query_logits = self.classifier(query_features, classifier_params)
            loss = self.loss(query_logits, query_examples['labels'])
            query_acc = self.accuracy_fn(query_logits, query_examples['labels'])

        return loss, query_acc

    def _text_encoding(self, examples, context_list):
        # Extract task-dependent adaptation parameters
        # Text encoding using the adaptted BERT model
        return self.encoder(context_list=context_list,
                                        **examples)[1]
    
    def _task_encoding(self, support_example, support_labels):
        #TODO: Better way for task encoding? Here we are simply using the average
        # representation of the support examples
        output = self.encoder(**support_example)[1]
        return torch.mean(output, dim=0)

    def _get_classifier_params(self, class_representations):
        """
        Processes the class representations and generated the linear classifier weights and biases.
        :return: Linear classifier weights and biases.
        """
        classifier_params = self.cl_adaptation_network(class_representations)
        return classifier_params

    def _build_class_reps(self, support_features, support_labels):
        class_representations = OrderedDict()
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(support_features,
                                                0,
                                                self._extract_class_indices(support_labels, c))
            class_rep = mean_pooling(class_features)
            class_representations[c.item()] = class_rep
        return class_representations
    
    ADAPT_NETWORK_WEIGHTS = 'adapt_network_weights.bin'
    CL_ADAPT_NETWORK_WEIGHTS = 'cl_adapt_network_weights.bin'
    CL_LINEAR_WEIGHTS = 'cl_linear_weights.bin'

    def from_pretrained(self, pretrained_path):
        self.adaptation_network.load_state_dict(torch.load(
            os.path.join(pretrained_path, self.ADAPT_NETWORK_WEIGHTS)
        ))
        self.cl_adaptation_network.load_state_dict(torch.load(
            os.path.join(pretrained_path, self.CL_ADAPT_NETWORK_WEIGHTS)
        ))
        self.cl_linear.load_state_dict(torch.load(
            os.path.join(pretrained_path, self.CL_LINEAR_WEIGHTS)
        ))

    def save_pretrained(self, output_dir):
        assert os.path.isdir(output_dir)
        
        self.encoder.save_pretrained(output_dir)
        torch.save(self.adaptation_network.state_dict(),
                   os.path.join(output_dir, self.ADAPT_NETWORK_WEIGHTS))
        torch.save(self.cl_adaptation_network.state_dict(),
                   os.path.join(output_dir, self.CL_ADAPT_NETWORK_WEIGHTS))
        torch.save(self.cl_linear.state_dict(),
                   os.path.join(output_dir, self.CL_LINEAR_WEIGHTS))

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def forward_old(self, film_param_dict_list, input_ids=None,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        """ Forward pass with FiLM adaptation.

        Args:
            film_param_dict_list (list<dict>): List of dictionary containing FiLM
                adaptaion parameters; one dictionary for each transformer layer.
        """


        outputs = self.bert(film_param_dict_list,
                            input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)