import os.path

import ipdb
from peft import set_peft_model_state_dict,get_peft_model_state_dict
from diffusers import FluxPipeline
from diffusers.training_utils import cast_training_params

def save_model_hook(models, weights, output_dir,wanted_model, accelerator,adapter_names):
    if accelerator.is_main_process:
        transformer_lora_layers_to_save = None
        for model in models:
            if isinstance(model, type(accelerator.unwrap_model(wanted_model))):
                transformer_lora_layers_to_save = {adapter_name: get_peft_model_state_dict(model,adapter_name=adapter_name) for adapter_name in adapter_names}
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()
        for adapter_name,lora in transformer_lora_layers_to_save.items():
            FluxPipeline.save_lora_weights(
                os.path.join(output_dir,adapter_name),
                transformer_lora_layers=lora,
            )


def load_model_hook(models, input_dir,wanted_model, accelerator,adapter_names):
    transformer_ = None
    while len(models) > 0:
        model = models.pop()
        if isinstance(model, type(accelerator.unwrap_model(wanted_model))):
            transformer_ = model
        else:
            raise ValueError(f"unexpected save model: {model.__class__}")

    lora_state_dict_list = []
    for adapter_name in adapter_names:
        lora_path = os.path.join(input_dir,adapter_name)
        lora_state_dict_list.append(FluxPipeline.lora_state_dict(lora_path))
    transformer_lora_state_dict_list = []
    for lora_state_dict in lora_state_dict_list:
        transformer_lora_state_dict_list.append({
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and "lora" in k
        })
    incompatible_keys = [set_peft_model_state_dict(transformer_, transformer_lora_state_dict_list[i], adapter_name=adapter_name) for i,adapter_name in enumerate(adapter_names)]
    if incompatible_keys is not None:
        # check only for unexpected keys
        unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        if unexpected_keys:
            accelerator.warning(
                f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                f" {unexpected_keys}. "
            )

    # Make sure the trainable params are in float32. This is again needed since the base models
    # are in `weight_dtype`. More details:
    # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
    if accelerator.mixed_precision == "fp16":
        models = [transformer_]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models)