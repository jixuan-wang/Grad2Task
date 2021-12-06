from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

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
from utils import linear_classifier, mean_pooling
from modeling_adaptation import (BottleneckAdapterAdaptContextShiftScaleAR, BottleneckAdapterAdaptContextWithTaskEmb, BottleneckAdapterAdaptContextWithTaskEmbCLS, BottleneckAdapterAdaptContextWithTaskEmbCLSShiftScale, BottleneckAdapterAdaptContextWithTaskEmbCLSShiftScaleAR, BottleneckAdapterAdaptInputOutput, BottleneckAdapterAdaptInputOutputWithTaskEmb, BottleneckAdapterWithoutTaskAdaptationWithShiftSacle, FilmAdaptationNetwork, BottleneckAdapter,
                                 LinearClassifierAdaptationNetwork)

################################################
# BERT output layer after self-attention layer #
class BertSelfOutputWithBNAdapter(BertSelfOutput):
    """ Fully connected layer after self-attention layer,
        with the bottleneck adapter layer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapter(config)
    
    def forward(self, hidden_states, input_tensor, params=None, return_bn_hidden=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Apply the adaptation layer.
        hidden_states, hidden_rep_tuple = self.adapter(hidden_states, params=params, return_hidden=return_bn_hidden)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, hidden_rep_tuple

class BertSelfOutputWithBNAdapterAdaptInputOutput(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptInputOutput(config)

class BertSelfOutputWithBNAdapterAdaptInputOutputTaskEmb(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptInputOutputWithTaskEmb(config)
    
    def regularization_term(self):
        return self.adapter.regularization_term()
class BertSelfOutputWithBNAdapterAdaptContextTaskEmb(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmb(config)
class BertSelfOutputWithBNAdapterAdaptContextTaskEmbCLS(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmbCLS(config)
class BertSelfOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScale(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmbCLSShiftScale(config)
class BertSelfOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmbCLSShiftScaleAR(config)

class BertSelfOutputWithBNAdapter_AdaptContextShiftScaleAR(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextShiftScaleAR(config)

class BertSelfOutputWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(BertSelfOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterWithoutTaskAdaptationWithShiftSacle(config)
################################################

#########################################
# BERT fully connected layer for output #
class BertOutputWithBNAdapter(BertOutput):
    """ Fully connected layer before output of transformer layer, 
        with the bottleneck adapter layer.
    """
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapter(config)

    def forward(self, hidden_states, input_tensor, params=None, return_bn_hidden=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Apply the adaptation layer.
        hidden_states, hidden_rep_tuple = self.adapter(hidden_states, params, return_hidden=return_bn_hidden)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, hidden_rep_tuple

class BertOutputWithBNAdapterAdaptInputOutput(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptInputOutput(config)
class BertOutputWithBNAdapterAdaptInputOutputTaskEmb(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptInputOutputWithTaskEmb(config)
        
    def regularization_term(self):
        return self.adapter.regularization_term()
class BertOutputWithBNAdapterAdaptContextTaskEmb(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmb(config)
class BertOutputWithBNAdapterAdaptContextTaskEmbCLS(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmbCLS(config)
class BertOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScale(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmbCLSShiftScale(config)
class BertOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextWithTaskEmbCLSShiftScaleAR(config)

class BertOutputWithBNAdapter_AdaptContextShiftScaleAR(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterAdaptContextShiftScaleAR(config)

class BertOutputWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(BertOutputWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.adapter = BottleneckAdapterWithoutTaskAdaptationWithShiftSacle(config)
    
#########################################

#############################
# BERT self-attention layer #
class BertAttentionWithBNAdapter(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapter(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                params=None, return_bn_hidden=False):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output, bn_hidden_tuple = self.output(self_outputs[0], hidden_states, params=params, return_bn_hidden=return_bn_hidden)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, bn_hidden_tuple

class BertAttentionWithBNAdapterAdaptInputOutput(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapterAdaptInputOutput(config)
class BertAttentionWithBNAdapterAdaptInputOutputTaskEmb(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapterAdaptInputOutputTaskEmb(config)

    def regularization_term(self):
        return self.output.regularization_term()
class BertAttentionWithBNAdapterAdaptContextTaskEmb(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapterAdaptContextTaskEmb(config)
class BertAttentionWithBNAdapterAdaptContextTaskEmbCLS(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapterAdaptContextTaskEmbCLS(config)
class BertAttentionWithBNAdapterAdaptContextTaskEmbCLSShiftScale(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScale(config)
class BertAttentionWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(config)
class BertAttentionWithBNAdapter_AdaptContextShiftScaleAR(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapter_AdaptContextShiftScaleAR(config)

class BertAttentionWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(BertAttentionWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.output = BertSelfOutputWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(config)

#############################

################################
#          BERT layer          #
class BertLayerWithBNAdapter(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapter(config)
        self.output = BertOutputWithBNAdapter(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                params_dict=None, return_bn_hidden=False):
        if params_dict is None:
            self_attention_outputs, bn_hidden_tuple1 = self.attention(hidden_states, attention_mask, head_mask, return_bn_hidden=return_bn_hidden)
        else:
            self_attention_outputs, bn_hidden_tuple1 = self.attention(hidden_states, attention_mask, head_mask,
                                                    params=params_dict['attention'],
                                                    return_bn_hidden=return_bn_hidden)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        if params_dict is None:
            layer_output, bn_hidden_tuple2 = self.output(intermediate_output, attention_output, return_bn_hidden=return_bn_hidden)
        else:
            layer_output, bn_hidden_tuple2 = self.output(intermediate_output, attention_output,
                                                            params=params_dict['output'],
                                                            return_bn_hidden=return_bn_hidden)
        outputs = (layer_output,) + outputs
        if return_bn_hidden:
            return outputs, bn_hidden_tuple1 + bn_hidden_tuple2
        else:
            return outputs, None

class BertLayerWithBNAdapterAdaptInputOutput(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapterAdaptInputOutput(config)
        self.output = BertOutputWithBNAdapterAdaptInputOutput(config)
class BertLayerWithBNAdapterAdaptInputOutputTaskEmb(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapterAdaptInputOutputTaskEmb(config)
        self.output = BertOutputWithBNAdapterAdaptInputOutputTaskEmb(config)

    def regularization_term(self):
        return self.output.regularization_term() + self.attention.regularization_term()
class BertLayerWithBNAdapterAdaptContextTaskEmb(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapterAdaptContextTaskEmb(config)
        self.output = BertOutputWithBNAdapterAdaptContextTaskEmb(config)
class BertLayerWithBNAdapterAdaptContextTaskEmbCLS(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapterAdaptContextTaskEmbCLS(config)
        self.output = BertOutputWithBNAdapterAdaptContextTaskEmbCLS(config)
class BertLayerWithBNAdapterAdaptContextTaskEmbCLSShiftScale(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapterAdaptContextTaskEmbCLSShiftScale(config)
        self.output = BertOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScale(config)
class BertLayerWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(config)
        self.output = BertOutputWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(config)

class BertLayerWithBNAdapter_AdaptContextShiftScaleAR(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapter_AdaptContextShiftScaleAR(config)
        self.output = BertOutputWithBNAdapter_AdaptContextShiftScaleAR(config)

class BertLayerWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(BertLayerWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(config)
        self.output = BertOutputWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(config)

################################

##################################
#          BERT encoder          #
class BertEncoderWithBNAdapter(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapter(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                params_dict_list=None, return_bn_hidden=False):
        all_hidden_states = ()
        all_attentions = ()
        bn_hidden_tuple_list = None
        if return_bn_hidden:
            bn_hidden_tuple_list = []
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if params_dict_list is None:
                layer_outputs, bn_hidden_tuple = layer_module(hidden_states, attention_mask, head_mask[i],
                                                                encoder_hidden_states, encoder_attention_mask,
                                                                return_bn_hidden=return_bn_hidden)
            else:
                layer_outputs, bn_hidden_tuple = layer_module(hidden_states, attention_mask, head_mask[i],
                                                                encoder_hidden_states, encoder_attention_mask,
                                                                params_dict=params_dict_list[i],
                                                                return_bn_hidden=return_bn_hidden)
            if bn_hidden_tuple is not None:
                bn_hidden_tuple_list.append(bn_hidden_tuple)

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
        return outputs, bn_hidden_tuple_list  # last-layer hidden state, (all hidden states), (all attentions)

class BertEncoderWithBNAdapterAdaptInputOutput(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapterAdaptInputOutput(config) for _ in range(config.num_hidden_layers)])
class BertEncoderWithBNAdapterAdaptInputOutputTaskEmb(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapterAdaptInputOutputTaskEmb(config) for _ in range(config.num_hidden_layers)])

    def regularization_term(self):
        l2_term = 0
        for l in self.layer:
            l2_term += l.regularization_term()
        return l2_term
class BertEncoderWithBNAdapterAdaptContextTaskEmb(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapterAdaptContextTaskEmb(config) for _ in range(config.num_hidden_layers)])
class BertEncoderWithBNAdapterAdaptContextTaskEmbCLS(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapterAdaptContextTaskEmbCLS(config) for _ in range(config.num_hidden_layers)])
class BertEncoderWithBNAdapterAdaptContextTaskEmbCLSShiftScale(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapterAdaptContextTaskEmbCLSShiftScale(config) for _ in range(config.num_hidden_layers)])
class BertEncoderWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(config) for _ in range(config.num_hidden_layers)])
class BertEncoderWithBNAdapter_AdaptContextShiftScaleAR(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapter_AdaptContextShiftScaleAR(config) for _ in range(config.num_hidden_layers)])

class BertEncoderWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(BertEncoderWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(config) for _ in range(config.num_hidden_layers)])
##################################

#######################
#      BERT Model     #
class BertModelWithBNAdapter(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapter(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                params_dict_list=None, return_bn_hidden=False):
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
        encoder_outputs, bn_hidden_tuple = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask,
                                       params_dict_list=params_dict_list,
                                       return_bn_hidden=return_bn_hidden)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        outputs += (bn_hidden_tuple,)
        return outputs   # sequence_output, pooled_output, (hidden_states), (attentions)

class BertModelWithBNAdapterAdaptInputOutput(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapterAdaptInputOutput(config)
        self.init_weights()

class BertModelWithBNAdapterAdaptInputOutputTaskEmb(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapterAdaptInputOutputTaskEmb(config)
        self.init_weights()
    
    def regularization_term(self):
        return self.encoder.regularization_term()

### Context adaptation
class BertModelWithBNAdapterAdaptContextTaskEmb(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapterAdaptContextTaskEmb(config)
        self.init_weights()

class BertModelWithBNAdapterAdaptContextTaskEmbCLS(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapterAdaptContextTaskEmbCLS(config)
        self.init_weights()

class BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScale(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapterAdaptContextTaskEmbCLSShiftScale(config)
        self.init_weights()
class BertModelWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapterAdaptContextTaskEmbCLSShiftScaleAR(config)
        self.init_weights()

class BertModelWithBNAdapter_AdaptContextShiftScaleAR(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapter_AdaptContextShiftScaleAR(config)
        self.init_weights()

class BertModelWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(BertModelWithBNAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithBNAdapter_WithoutTaskAdaptationWithShiftSacle(config)
        self.init_weights()

#######################

####################### END ####################### 

class BertWithBNAdapterForSequenceClassification(BertForSequenceClassification):
    """ BERT with the bottleneck adapter for sequence classification tasks. """
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelWithBNAdapter(config)

        for n, p in self.bert.named_parameters():
            if not 'adapter' in n:
                p.requires_grad = False
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                params_dict_list=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            params_dict_list=params_dict_list)

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

#########################################
# BERT components with the FiLM adapter #
#########################################
class BertSelfAttentionWithFilmAdapter(BertSelfAttention):
    """ Self-attention layer with the FiLM adapter layer. """
    def __init__(self, config):
        super().__init__(config)
    
    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return x * gamma + beta

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                film_param_dict=None):
        """ Forward pass with the FiLM adaptor. One gamma / beta set for each
            attention head.

        Args:
            param_dict (dict): gamma / beta for query, key and value of all 
            attention heads.
            hidden_states ([type]): [description]
            attention_mask ([type], optional): [description]. Defaults to None.
            head_mask ([type], optional): [description]. Defaults to None.
            encoder_hidden_states ([type], optional): [description]. Defaults to None.
            encoder_attention_mask ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if film_param_dict is not None:
            # Apply the FiLM layer
            # query_layer: batch_size x num_attention_head x seq_length x hidden_dim
            key_layer = self._film(key_layer, film_param_dict['gamma_k'], film_param_dict['beta_k'])
            value_layer = self._film(value_layer, film_param_dict['gamma_v'], film_param_dict['beta_v'])
            query_layer = self._film(query_layer, film_param_dict['gamma_q'], film_param_dict['beta_q'])

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutputWithFilmAdapter(BertSelfOutput):
    """ Fully connected layer after self-attention layer, 
        with the FiLM adapter layer.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, hidden_states, input_tensor, film_param_dict=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if film_param_dict is not None:
            # Apply the adaptation layer
            hidden_states = self._film(hidden_states,
                                       film_param_dict['gamma_self_linear'],
                                       film_param_dict['beta_self_linear'])
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    def _film(self, x, gamma, beta):
        gamma = gamma[None, None, :]
        beta = beta[None, None, :]
        return x * gamma + beta

class BertOutputWithFilmAdapter(BertOutput):
    """ Fully connected layer before output of transformer layer, 
        with the FiLM adapter layer.
    """
    def __init__(self, config):
        super().__init__(config)

    def forward(self, hidden_states, input_tensor, film_param_dict=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if film_param_dict is not None:
            # Apply the adaptation layer 
            hidden_states = self._film(hidden_states,
                                       film_param_dict['gamma_out_linear'],
                                       film_param_dict['beta_out_linear'])
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def _film(self, x, gamma, beta):
        gamma = gamma[None, None, :]
        beta = beta[None, None, :]
        return x * gamma + beta

class BertAttentionWithFilmAdapter(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionWithFilmAdapter(config)
        self.output = BertSelfOutputWithFilmAdapter(config)

    def forward(self, hidden_states, attention_mask=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                film_param_dict=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
                                 film_param_dict=film_param_dict)
        attention_output = self.output(self_outputs[0], hidden_states,
                                       film_param_dict=film_param_dict)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertLayerWithFilmAdapter(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionWithFilmAdapter(config)
        self.output = BertOutputWithFilmAdapter(config)

    def forward(self, hidden_states, attention_mask=None,
                head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                film_param_dict=None):
        # Self attention layer with film parameters
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,
                                                film_param_dict=film_param_dict)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        # Output layer with film parameters
        layer_output = self.output(intermediate_output, attention_output,
                                   film_param_dict=film_param_dict)
        outputs = (layer_output,) + outputs
        return outputs

class BertEncoderWithFilmAdapter(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList([BertLayerWithFilmAdapter(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                film_param_dict_list=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Apply film adaptation
            if film_param_dict_list is not None:
                # Fetch film parameters for the current layer
                film_param_dict = film_param_dict_list[i]
                # Transformer layer with film parameters
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i],
                                            encoder_hidden_states, encoder_attention_mask,
                                            film_param_dict=film_param_dict)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i],
                                                encoder_hidden_states, encoder_attention_mask)
                
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

class BertModelWithFilmAdapter(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoderWithFilmAdapter(config)
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,
                film_param_dict_list=None, **kwargs):
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
        # BERT encoder with film parameters
        encoder_outputs = self.encoder(embedding_output,
                                        attention_mask=extended_attention_mask,
                                        head_mask=head_mask,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_extended_attention_mask,
                                        film_param_dict_list=film_param_dict_list)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class BertWithFilmAdapterForMetatraining(nn.Module):
    """ BERT with the FiLM adapter layer for meta-training on various tasks. """ 
    def __init__(self, args, bert_config):
        super().__init__()
        if args.adapter_type == 'film':
            self.encoder = BertModelWithFilmAdapter.from_pretrained(
                            args.model_name_or_path,
                            from_tf=bool('.ckpt' in args.model_name_or_path),
                            config=bert_config,
                            cache_dir=args.cache_dir if args.cache_dir else None)
            self.adaptation_network = FilmAdaptationNetwork(
                                        task_rep_dim=bert_config.hidden_size,
                                        num_target_layer=bert_config.num_hidden_layers,
                                        num_attention_head=bert_config.num_attention_heads)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # Freeze the parameters of the text encoder (BERT)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
        self.classifier = linear_classifier
        # Adapation network for the linear classifier
        self.cl_adaptation_network = LinearClassifierAdaptationNetwork(bert_config.hidden_size)

        self.task_representation = None
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        self.total_iterations = args.num_training_iterations

    def forward(self, support_examples, query_examples):
        # Extract task representaion
        self.task_representation = self._task_encoding(support_examples, support_examples['labels'])
        # Text encoding
        support_features, query_features = self._text_encoding(support_examples, query_examples)
        # Build task-dependent linear classifier
        self._build_class_reps(support_features, support_examples['labels'])
        classifier_params = self._get_classifier_params()
        # Classify
        sample_logits = self.classifier(query_features, classifier_params)
        self.class_representations.clear()
        return sample_logits

    def _text_encoding(self, support_examples, query_examples):
        # Extract task-dependent adaptation parameters
        # self.feature_extractor_film_params = self.adaptation_network(self.task_representation)
        self.feature_extractor_film_params = None #FIXME !!!!!!!!!!!
        # Text encoding using the adaptted BERT model
        support_features = self.encoder(film_param_dict_list=self.feature_extractor_film_params,
                                        **support_examples)[1]
        query_features = self.encoder(film_param_dict_list=self.feature_extractor_film_params,
                                      **query_examples)[1]
        return support_features, query_features
    
    def _task_encoding(self, support_example, support_labels):
        #TODO: Better way for task encoding? Here we are simply using the average
        # representation of the support examples
        output = self.encoder(**support_example)[1]
        return torch.mean(output, dim=0)

    def _get_classifier_params(self):
        """
        Processes the class representations and generated the linear classifier weights and biases.
        :return: Linear classifier weights and biases.
        """
        classifier_params = self.cl_adaptation_network(self.class_representations)
        return classifier_params

    def _build_class_reps(self, support_features, support_labels):
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(support_features,
                                                0,
                                                self._extract_class_indices(support_labels, c))
            class_rep = mean_pooling(class_features)
            self.class_representations[c.item()] = class_rep
    
    def from_pretrained(self, pretrained_path):
        self.adaptation_network.load_state_dict(torch.load(
            os.path.join(pretrained_path, self.ADAPT_NETWORK_WEIGHTS)
        ))
        self.cl_adaptation_network.load_state_dict(torch.load(
            os.path.join(pretrained_path, self.CL_ADAPT_NETWORK_WEIGHTS)
        ))

    ADAPT_NETWORK_WEIGHTS = 'adapt_network_weights.bin'
    CL_ADAPT_NETWORK_WEIGHTS = 'cl_adapt_network_weights.bin'
    def save_pretrained(self, output_dir):
        assert os.path.isdir(output_dir)
        
        self.encoder.save_pretrained(output_dir)
        torch.save(self.adaptation_network.state_dict(),
                   os.path.join(output_dir, self.ADAPT_NETWORK_WEIGHTS))
        torch.save(self.cl_adaptation_network.state_dict(),
                   os.path.join(output_dir, self.CL_ADAPT_NETWORK_WEIGHTS))

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

class BertWithFilmAdapterForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertWithFilmAdapterForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert_config = config

        self.bert = BertModelWithFilmAdapter(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        # Freeze the parameters of the text encoder (BERT)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.init_weights()
        self.init_film_weights()

    def init_film_weights(self):
        for i in range(self.bert_config.num_hidden_layers):
            for name in ['k', 'q', 'v'] + ['self_linear', 'out_linear']:
                num_param = 1 if 'linear' in name else self.bert_config.num_attention_heads 
                gamma_param = nn.Parameter(
                        torch.nn.init.normal_(torch.empty(num_param), 1, 0.001),
                        requires_grad=True
                )
                self.register_parameter(f'gamma_{name}_{i}', gamma_param)
    
                beta_param = nn.Parameter(
                        torch.nn.init.normal_(torch.empty(num_param), 0, 0.001),
                        requires_grad=True
                )
                self.register_parameter(f'beta_{name}_{i}', gamma_param)
    @property
    def film_param_dict_list(self):
        film_param_dict_list = []
        for i in range(self.bert_config.num_hidden_layers):
            film_param_dict = {}
            for name in ['k', 'q', 'v'] + ['self_linear', 'out_linear']:
                for f_name in ['gamma', 'beta']:
                    film_param_dict[f'{f_name}_{name}'] = getattr(
                        self, f'{f_name}_{name}_{i}'
                    ) 
            film_param_dict_list.append(film_param_dict)
        return film_param_dict_list
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            film_param_dict_list=self.film_param_dict_list)

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