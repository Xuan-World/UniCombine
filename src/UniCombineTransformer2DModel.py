import copy
from diffusers.configuration_utils import register_to_config
from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
import torch
from diffusers.models.transformers.transformer_flux import (
    FluxTransformer2DModel,
    Transformer2DModelOutput,
)
from diffusers.utils import unscale_lora_layers,is_torch_version,USE_PEFT_BACKEND,scale_lora_layers,logging
from .lora_switching_module import enable_lora, module_active_adapters
from .UniCombineTransformerBlock import block_forward,single_block_forward

logger = logging.get_logger(__name__)
class UniCombineTransformer2DModel(FluxTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__(patch_size,
                         in_channels,
                         out_channels,
                         num_layers,
                         num_single_layers,
                         attention_head_dim,
                         num_attention_heads,
                         joint_attention_dim,
                         pooled_projection_dim,
                         guidance_embeds,
                         axes_dims_rope)

    def forward(self,
                hidden_states: torch.Tensor,
                condition_latents: List[torch.Tensor],
                condition_ids: List[torch.Tensor],
                condition_type_ids: List[torch.Tensor],
                condition_types: List[str],
                model_config: Optional[Dict[str, Any]] = {},
                return_condition_latents: bool = False,
                c_t=0,
                encoder_hidden_states: torch.Tensor = None,
                pooled_projections: torch.Tensor = None,
                timestep: torch.LongTensor = None,
                img_ids: torch.Tensor = None,
                txt_ids: torch.Tensor = None,
                guidance: torch.Tensor = None,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                controlnet_block_samples=None,
                controlnet_single_block_samples=None,
                return_dict: bool = True,
                controlnet_blocks_repeat: bool = False,
            ) -> tuple[Any, None] | tuple[Any, Any | None] | Transformer2DModelOutput:
        use_condition = condition_latents is not None

        # lora scale
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # MAYBE a conflict when loading multi-loras, seems to weight them together. Weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        # hidden_state proj
        with enable_lora([self.x_embedder],[item for item in module_active_adapters(self.x_embedder) if item not in condition_types]):
            hidden_states = self.x_embedder(hidden_states)
        # condition proj
        if use_condition:
            condition_latents = copy.deepcopy(condition_latents)
            for i, cond_type in enumerate(condition_types):
                with enable_lora([self.x_embedder],[cond_type]):
                    condition_latents[i] = self.x_embedder(condition_latents[i])

        # text_embedding proj
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # prepare for timestep and guidance value
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        # computing the time_poolingtext_guidance embedding for the text branch and the denoising branch
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        # computing the time_poolingtext_guidance embedding for the conditional branches
        cond_temb = (
            self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, pooled_projections)
            if guidance is None
            else self.time_text_embed(torch.ones_like(timestep) * c_t * 1000, guidance, pooled_projections)
        )

        # not use in this version
        if hasattr(self, "cond_type_embed") and condition_type_ids is not None:
            cond_type_proj = self.time_text_embed.time_proj(condition_type_ids[0])
            cond_type_emb = self.cond_type_embed(cond_type_proj.to(dtype=cond_temb.dtype))
            cond_temb = cond_temb + cond_type_emb

        # Rotary Positional Embedding
        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = tuple(i.to(self.dtype) for i in self.pos_embed(ids))
        cond_rotary_embs = []
        if use_condition:
            for cond_id in condition_ids:
                cond_rotary_embs.append(tuple(i.to(self.dtype) for i in self.pos_embed(cond_id)))

        # process in mm-DiT_block
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states, condition_latents = block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                condition_latents= condition_latents if use_condition else None,
                condition_types = condition_types if use_condition else None,
                temb=temb,
                cond_temb=cond_temb if use_condition else None,
                image_rotary_emb=image_rotary_emb,
                cond_rotary_embs=cond_rotary_embs if use_condition else None,
            )
            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = (hidden_states + controlnet_block_samples[index_block // interval_control])

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # process in single-DiT_block
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states, condition_latents = single_block_forward(
                block,
                model_config=model_config,
                hidden_states=hidden_states,
                condition_latents= condition_latents if use_condition else None,
                condition_types=condition_types if use_condition else None,
                temb=temb,
                cond_temb= cond_temb if use_condition else None,
                image_rotary_emb=image_rotary_emb,
                cond_rotary_embs= cond_rotary_embs if use_condition else None,
            )
            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]+ controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]

        hidden_states = self.norm_out(hidden_states, temb).to(self.dtype)
        output = self.proj_out(hidden_states)
        if return_condition_latents:
            condition_latents = [ self.proj_out(self.norm_out(i, cond_temb)) if use_condition else None for i in condition_latents]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,None) if not return_condition_latents else (output, condition_latents)

        return Transformer2DModelOutput(sample=output)
