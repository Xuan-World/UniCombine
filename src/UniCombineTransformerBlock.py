import ipdb
import torch
from typing import List, Optional, Dict, Any
from torch import FloatTensor, Tensor
from diffusers.models.attention_processor import Attention, F
from .lora_switching_module import enable_lora,module_active_adapters
from diffusers.models.embeddings import apply_rotary_emb

def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    condition_types: List[str],
    encoder_hidden_states: torch.FloatTensor = None,
    condition_latents: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    cond_rotary_embs: Optional[List[torch.Tensor]] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> tuple[Any, Any, list[FloatTensor] | None] | tuple[Any, Any] | tuple[Tensor, Tensor] | Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    # base_key / base_value: [text,noise] if encoder_hidden_states is not None else [noise]
    with enable_lora([attn.to_q, attn.to_k, attn.to_v], [item for item in module_active_adapters(attn.to_q) if item not in condition_types]):
        base_key = attn.to_k(hidden_states)
        base_value = attn.to_v(hidden_states)
        query = attn.to_q(hidden_states)

    inner_dim = query.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    base_key = base_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    base_value = base_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        base_key = attn.norm_k(base_key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        seq_len  =  seq_len + encoder_hidden_states.shape[1]
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # concat the text embedding and noise embedding
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        base_key = torch.cat([encoder_hidden_states_key_proj, base_key], dim=2)
        base_value = torch.cat([encoder_hidden_states_value_proj, base_value], dim=2)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        base_key = apply_rotary_emb(base_key, image_rotary_emb)

    condition_latents_output_list = []

    key = base_key
    value = base_value
    if condition_latents is not None and len(condition_latents) > 0:
        for i, cond_type in enumerate(condition_types):
            with enable_lora([attn.to_q,attn.to_k, attn.to_v], [cond_type]):
                cond_query = attn.to_q(condition_latents[i])
                cond_key = attn.to_k(condition_latents[i])
                cond_value = attn.to_v(condition_latents[i])
            cond_query = cond_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cond_key = cond_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cond_value = cond_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                cond_query = attn.norm_q(cond_query)
            if attn.norm_k is not None:
                cond_key = attn.norm_k(cond_key)

            if cond_rotary_embs is not None:
                cond_query = apply_rotary_emb(cond_query, cond_rotary_embs[i])
                cond_key = apply_rotary_emb(cond_key, cond_rotary_embs[i])

            key = torch.cat([key, cond_key], dim=2)
            value = torch.cat([value, cond_value], dim=2)
            mix_cond_key = torch.cat([base_key, cond_key], dim=2)
            mix_cond_value = torch.cat([base_value, cond_value], dim=2)

            condition_latents_output = F.scaled_dot_product_attention(
                cond_query, mix_cond_key, mix_cond_value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
            )
            condition_latents_output = condition_latents_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            condition_latents_output_list.append(condition_latents_output)

    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )

    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )
        with enable_lora([attn.to_out[0]],  [item for item in module_active_adapters(attn.to_out[0]) if item not in condition_types]):
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if condition_latents is not None and len(condition_latents) > 0:
            for i, cond_type in enumerate(condition_types):
                with enable_lora([attn.to_out[0]], [cond_type]):
                    condition_latents_output_list[i] = attn.to_out[0](condition_latents_output_list[i])
                    condition_latents_output_list[i] = attn.to_out[1](condition_latents_output_list[i])
        return hidden_states, encoder_hidden_states, condition_latents_output_list

    else:
        return hidden_states, condition_latents_output_list



def block_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            condition_latents: List[torch.Tensor] = None,
            temb: torch.Tensor = None,
            cond_temb: List[torch.Tensor] = None,
            cond_rotary_embs=None,
            image_rotary_emb=None,
            condition_types: List[str]=None,
            model_config: Optional[Dict[str, Any]] = {},
    ):
    use_cond = condition_latents is not None and len(condition_latents) > 0
    # norm :  hidden_states->norm_hidden_states, encoder_hidden_states->norm_encoder_hidden_states, condition_latents->norm_condition_latent_list
    with enable_lora([self.norm1.linear], [item for item in module_active_adapters(self.norm1.linear) if item not in condition_types]):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )
    norm_condition_latent_list = []
    cond_gate_msa_list = []
    cond_shift_mlp_list = []
    cond_scale_mlp_list = []
    cond_gate_mlp_list = []
    if use_cond:
        for i, cond_type in enumerate(condition_types):
            with enable_lora([self.norm1.linear],[cond_type]):
                norm_condition_latent,cond_gate_msa,cond_shift_mlp,cond_scale_mlp,cond_gate_mlp,= self.norm1(
                    condition_latents[i], emb=cond_temb
                )
            norm_condition_latent_list.append(norm_condition_latent)
            cond_gate_msa_list.append(cond_gate_msa)
            cond_shift_mlp_list.append(cond_shift_mlp)
            cond_scale_mlp_list.append(cond_scale_mlp)
            cond_gate_mlp_list.append(cond_gate_mlp)

    # Attention.  attn_output, context_attn_output, cond_attn_output_list
    attn_output, context_attn_output,cond_attn_output_list = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        condition_types=condition_types,
        encoder_hidden_states=norm_encoder_hidden_states,
        condition_latents=norm_condition_latent_list if use_cond else None,
        image_rotary_emb=image_rotary_emb,
        cond_rotary_embs=cond_rotary_embs if use_cond else None,
    )
    # Process attention outputs. Gate + Resnet. hidden_states, encoder_hidden_states, condition_latents
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output
    # 3. condition_latents
    if use_cond:
        for i, cond_type in enumerate(condition_types):
            cond_attn_output_list[i] = cond_gate_msa_list[i].unsqueeze(1) * cond_attn_output_list[i]
            condition_latents[i] = condition_latents[i] + cond_attn_output_list[i]
            if model_config.get("add_cond_attn", False):
                hidden_states += cond_attn_output_list[i]

    # LayerNorm + Scaling + Shift.
    # hidden_states->norm_hidden_states, encoder_hidden_states->norm_encoder_hidden_states, condition_latents->norm_condition_latent_list
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    # 3. condition_latents
    if use_cond:
        for i, cond_type in enumerate(condition_types):
            norm_condition_latent_list[i] = self.norm2(condition_latents[i])
            norm_condition_latent_list[i] = norm_condition_latent_list[i] * (1 + cond_scale_mlp_list[i][:, None]) + cond_shift_mlp_list[i][:, None]

    # MLP Feed-forward + Gate
    # 1. hidden_states
    with enable_lora([self.ff.net[2]], [item for item in module_active_adapters(self.ff.net[2]) if item not in condition_types]):
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)
    # 2. encoder_hidden_states
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_encoder_hidden_states)
    # 3. condition_latents
    if use_cond:
        for i, cond_type in enumerate(condition_types):
            with enable_lora([self.ff.net[2]], [cond_type]):
                condition_latents[i] = condition_latents[i] + cond_gate_mlp_list[i].unsqueeze(1) * self.ff(norm_condition_latent_list[i])

    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states, condition_latents if use_cond else None


def single_block_forward(
        self,
        hidden_states: torch.Tensor,
        condition_latents: List[torch.Tensor] = None,
        temb: torch.Tensor = None,
        cond_temb: List[torch.Tensor] = None,
        image_rotary_emb=None,
        cond_rotary_embs=None,
        condition_types: List[str] = None,
        model_config: Optional[Dict[str, Any]] = {},
    ):
    using_cond = condition_latents is not None and len(condition_latents) > 0
    with enable_lora([self.norm.linear,self.proj_mlp],[item for item in module_active_adapters(self.norm.linear) if item not in condition_types]):
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    norm_condition_latent_list = []
    mlp_condition_latent_list = []
    cond_gate_list = []

    if using_cond:
        for i, cond_type in enumerate(condition_types):
            with enable_lora([self.norm.linear, self.proj_mlp],[cond_type]):
                norm_condition_latents, cond_gate = self.norm(condition_latents[i], emb=cond_temb)
                mlp_condition_latents = self.act_mlp(self.proj_mlp(norm_condition_latents))
            norm_condition_latent_list.append(norm_condition_latents)
            mlp_condition_latent_list.append(mlp_condition_latents)
            cond_gate_list.append(cond_gate)

    attn_output, cond_attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        condition_types= condition_types,
        image_rotary_emb=image_rotary_emb,
        **(
            {
                "condition_latents": norm_condition_latent_list,
                "cond_rotary_embs": cond_rotary_embs if using_cond else None,
            }
            if using_cond
            else {}
        ),
    )
    with enable_lora([self.proj_out], [item for item in module_active_adapters(self.proj_out) if item not in condition_types]):
        hidden_states = hidden_states + gate.unsqueeze(1) * self.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
    if using_cond:
        for i, cond_type in enumerate(condition_types):
            with enable_lora([self.proj_out],[cond_type]):
                attn_mlp_condition_latents = torch.cat([cond_attn_output[i], mlp_condition_latent_list[i]], dim=2)
                attn_mlp_condition_latents = cond_gate_list[i].unsqueeze(1) * self.proj_out(attn_mlp_condition_latents)
                condition_latents[i] = condition_latents[i] + attn_mlp_condition_latents

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return (hidden_states,None) if not using_cond else (hidden_states, condition_latents)

