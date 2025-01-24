# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch MBART model."""
from functools import partial
import random
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.mbart.modeling_mbart import (
    MBartAttention,
    MBartDecoder,
    MBartForCausalLM,
    _expand_mask,
    logger,
)
import xformers.ops as xops


def apply_fast_mbart_decoder(model: MBartForCausalLM) -> None:
    for module in model.model.modules():
        if isinstance(module, MBartDecoder):
            module.forward = partial(mbart_decoder_fast_forward, module)
        if isinstance(module, MBartAttention):
            module.forward = partial(mbart_attention_fast_forward, module)


def mbart_attention_fast_forward(
    mbart_attention_module: MBartAttention,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = mbart_attention_module.q_proj(hidden_states)
    # get key, value proj
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if (
        is_cross_attention
        and past_key_value is not None
        and past_key_value[0].shape[2] == key_value_states.shape[1]
    ):
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = mbart_attention_module._shape(
            mbart_attention_module.k_proj(key_value_states), -1, bsz
        )
        value_states = mbart_attention_module._shape(
            mbart_attention_module.v_proj(key_value_states), -1, bsz
        )
    elif past_key_value is not None:
        # reuse k, v, self_attention
        key_states = mbart_attention_module._shape(
            mbart_attention_module.k_proj(hidden_states), -1, bsz
        )
        value_states = mbart_attention_module._shape(
            mbart_attention_module.v_proj(hidden_states), -1, bsz
        )
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        key_states = mbart_attention_module._shape(
            mbart_attention_module.k_proj(hidden_states), -1, bsz
        )
        value_states = mbart_attention_module._shape(
            mbart_attention_module.v_proj(hidden_states), -1, bsz
        )

    if mbart_attention_module.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    proj_shape = (
        bsz,
        mbart_attention_module.num_heads,
        -1,
        mbart_attention_module.head_dim,
    )
    query_states = (
        mbart_attention_module._shape(query_states, tgt_len, bsz)
        .view(*proj_shape)
        .transpose(1, 2)
    )
    key_states = key_states.reshape(*proj_shape).transpose(1, 2)
    value_states = value_states.reshape(*proj_shape).transpose(1, 2)

    # xformers memory efficient attention with no mask or causal mask
    attn_bias = None
    if attention_mask is not None:
        seq_len = attention_mask.shape[3]
        if seq_len % 8 != 0:
            strided_seq_len = (seq_len // 8 + 1) * 8
            attn_bias = torch.zeros(
                attention_mask.size()[:-1] + (strided_seq_len,),
                device=attention_mask.device,
                dtype=query_states.dtype,
            )
            attn_bias[:, :, :, :seq_len] = attention_mask
            attn_bias = attn_bias[:, :, :, :seq_len]
        else:
            attn_bias = attention_mask
        if attn_bias.shape[1] == 1:
            attn_bias = attn_bias.expand(
                -1, query_states.shape[2], -1, -1
            )  # expand by num_heads

    # if not is_cross_attention:
    #     if mbart_attention_module.training:
    #         attn_bias = xops.LowerTriangularMask()
    attn_output = xops.memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        p=mbart_attention_module.dropout if mbart_attention_module.training else 0.0,
        attn_bias=attn_bias,
    )

    if attn_output.size() != (
        bsz,
        tgt_len,
        mbart_attention_module.num_heads,
        mbart_attention_module.head_dim,
    ):
        raise ValueError(
            "`attn_output` should be of size "
            f"{(bsz, tgt_len, mbart_attention_module.num_heads, mbart_attention_module.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.view(
        bsz, tgt_len, mbart_attention_module.num_heads, mbart_attention_module.head_dim
    )
    # attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, mbart_attention_module.embed_dim)

    attn_output = mbart_attention_module.out_proj(attn_output)

    return attn_output, None, past_key_value


def mbart_decoder_fast_forward(
    mbart_decoder_module: MBartDecoder,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
            provide it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
            selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
            cross-attention on hidden heads. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*,
            returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
            cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
            all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
            shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
            `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
            control over how to convert `input_ids` indices into associated vectors than the model's internal
            embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else mbart_decoder_module.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else mbart_decoder_module.config.output_hidden_states
    )
    use_cache = (
        use_cache if use_cache is not None else mbart_decoder_module.config.use_cache
    )
    return_dict = (
        return_dict
        if return_dict is not None
        else mbart_decoder_module.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input = input_ids
        input_shape = input.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    # past_key_values_length
    past_key_values_length = (
        past_key_values[0][0].shape[2] if past_key_values is not None else 0
    )

    if inputs_embeds is None:
        inputs_embeds = (
            mbart_decoder_module.embed_tokens(input_ids)
            * mbart_decoder_module.embed_scale
        )

    # No need to make attention_mask for fast attention -> revived to allow custom attention masking
    attention_mask = mbart_decoder_module._prepare_decoder_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

    # embed positions
    positions = mbart_decoder_module.embed_positions(input, past_key_values_length)

    hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
    hidden_states = mbart_decoder_module.layernorm_embedding(hidden_states)

    hidden_states = nn.functional.dropout(
        hidden_states,
        p=mbart_decoder_module.dropout,
        training=mbart_decoder_module.training,
    )

    if mbart_decoder_module.gradient_checkpointing and mbart_decoder_module.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = (
        () if (output_attentions and encoder_hidden_states is not None) else None
    )
    next_decoder_cache = () if use_cache else None

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip(
        [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
    ):
        if attn_mask is not None:
            if attn_mask.size()[0] != len(mbart_decoder_module.layers):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(mbart_decoder_module.layers)} layers, but it is for"
                    f" {attn_mask.size()[0]}."
                )
    for idx, decoder_layer in enumerate(mbart_decoder_module.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        dropout_probability = random.uniform(0, 1)
        if mbart_decoder_module.training and (
            dropout_probability < mbart_decoder_module.layerdrop
        ):
            continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if (
            mbart_decoder_module.gradient_checkpointing
            and mbart_decoder_module.training
        ):

            def create_custom_forward(module: nn.Module) -> Callable:
                def custom_forward(*inputs: Any) -> Any:
                    # None for past_key_value
                    return module(*inputs, output_attentions, use_cache)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                head_mask[idx] if head_mask is not None else None,
                cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

    hidden_states = mbart_decoder_module.layer_norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        cross_attentions=all_cross_attentions,
    )
