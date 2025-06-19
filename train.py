import sys,os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import argparse
import copy
import logging
import math
import os
from contextlib import contextmanager
import functools
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from src.hook import save_model_hook,load_model_hook
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from src.dataloader import get_dataset,prepare_dataset,collate_fn
if is_wandb_available():
    pass
from src.text_encoder import encode_prompt
from datetime import datetime
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@contextmanager
def preserve_requires_grad(model):
    # 备份 requires_grad 状态
    requires_grad_backup = {name: param.requires_grad for name, param in model.named_parameters()}
    yield
    # 恢复 requires_grad 状态
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_backup[name]
def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument( "--pretrained_model_name_or_path",type=str,default="ckpt/FLUX.1-schnell")
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--work_dir",type=str,default="output/train_result",)
    parser.add_argument("--output_denoising_lora",type=str,default="depth_canny_union",)
    parser.add_argument("--pretrained_condition_lora_dir",type=str,default="ckpt/Condition_LoRA",)
    parser.add_argument("--training_adapter",type=str,default="ckpt/FLUX.1-schnell-training-adapter",)
    parser.add_argument("--checkpointing_steps",type=int,default=1,)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,)
    parser.add_argument("--rank",type=int,default=4,help="The dimension of the LoRA rank.")

    parser.add_argument("--dataset_name",type=str,default=[
            "dataset/split_SubjectSpatial200K/train",
            "dataset/split_SubjectSpatial200K/Collection3/train",
        ],
    )
    parser.add_argument("--image_column", type=str, default="image",)
    parser.add_argument("--bbox_column",type=str,default="bbox",)
    parser.add_argument("--canny_column",type=str,default="canny",)
    parser.add_argument("--depth_column",type=str,default="depth",)
    parser.add_argument("--condition_types",type=str,nargs='+',default=["depth","canny"],)

    parser.add_argument("--max_sequence_length",type=int,default=512,help="Maximum sequence length to use with with the T5 text encoder")
    parser.add_argument("--mixed_precision",type=str,default="bf16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--cache_dir",type=str,default="cache",)
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=30000,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=2)

    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--scale_lr",action="store_true",default=False,)
    parser.add_argument("--lr_scheduler",type=str,default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500,)
    parser.add_argument("--weighting_scheme",type=str,default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument("--dataloader_num_workers",type=int,default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True)

    args = parser.parse_args()
    args.revision = None
    args.variant = None
    args.work_dir = os.path.join(args.work_dir,datetime.now().strftime('%y_%m_%d-%H:%M'))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    text_encoder_one = text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device, dtype=weight_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)


    transformer = UniCombineTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant
    ).to(accelerator.device, dtype=weight_dtype)
    # load lora !!!!!
    lora_names = args.condition_types
    for condition_type in lora_names:
        transformer.load_lora_adapter(f"{args.pretrained_condition_lora_dir}/{condition_type}.safetensors", adapter_name=condition_type)

    transformer.load_lora_adapter(f"{args.training_adapter}/pytorch_lora_weights.safetensors", adapter_name="schnell_assistant")

    logger.info("All models loaded successfully")
    # freeze parameters of models to save more memory
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    logger.info("All models keeps requires_grad = False")

    single_transformer_blocks_lora = [
        f"single_transformer_blocks.{i}.proj_out"
        for i in range(len(transformer.single_transformer_blocks))
    ] + [
        f"single_transformer_blocks.{i}.proj_mlp"
        for i in range(len(transformer.single_transformer_blocks))
    ]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "x_embedder",
            "norm1.linear",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.2",
            "norm.linear",
        ]+single_transformer_blocks_lora,
    )
    transformer.add_adapter(transformer_lora_config,adapter_name=args.output_denoising_lora)
    logger.info(f"Trainable lora: {args.output_denoising_lora} is loaded successfully")
    # hook
    accelerator.register_save_state_pre_hook(functools.partial(save_model_hook,wanted_model=transformer,accelerator=accelerator,adapter_names=[args.output_denoising_lora]))
    accelerator.register_load_state_pre_hook(functools.partial(load_model_hook,wanted_model=transformer,accelerator=accelerator,adapter_names=[args.output_denoising_lora]))
    logger.info("Hooks for save and load is ok.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("Optimizer initialized successfully.")

    # Preprocessing the datasets.
    train_dataset = get_dataset(args)
    train_dataset = prepare_dataset(train_dataset, vae_scale_factor, accelerator, args)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    logger.info("Training dataset and Dataloader initialized successfully.")

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids


    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    logger.info(f"lr_scheduler:{args.lr_scheduler} initialized successfully.")

    with preserve_requires_grad(transformer):
        transformer.set_adapters([i for i in lora_names] + [args.output_denoising_lora] + ['schnell_assistant'])
    logger.info(f"Set Adapters:{[i for i in lora_names] + [args.output_denoising_lora] + ['schnell_assistant']}")

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("UniCombine", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.work_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                prompts = batch["descriptions"]
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )
                # 1.1 Convert images to latent space.
                latent_image = encode_images(pixels=batch["pixel_values"],vae=vae,weight_dtype=weight_dtype)
                # 1.2 Get positional id.
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latent_image.shape[0],
                    latent_image.shape[2] // 2,
                    latent_image.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # 2.1 Convert Conditions to latent space list.
                # 2.2 Get Conditions positional id list.
                # 2.3 Get Conditions types string list.
                # (bs, cond_num, c, h, w) -> [cond_num, (bs, c, h ,w)]
                condition_latents = list(torch.unbind(batch["condition_latents"], dim=1))
                # [cond_num, (len ,3) ]
                condition_ids = []
                # [cond_num]
                condition_types = batch["condition_types"][0]
                for i,images_per_condition in enumerate(condition_latents):
                    # i means condition No.i.
                    # images_per_condition = (bs, c, h ,w)
                    images_per_condition = encode_images(pixels=images_per_condition,vae=vae,weight_dtype=weight_dtype)
                    cond_ids = FluxPipeline._prepare_latent_image_ids(
                        images_per_condition.shape[0],
                        images_per_condition.shape[2] // 2,
                        images_per_condition.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    if condition_types[i] == "subject":
                        cond_ids[:, 2] += images_per_condition.shape[2] // 2
                    condition_ids.append(cond_ids)
                    condition_latents[i] = images_per_condition

                # 3 Sample noise that we'll add to the latents
                noise = torch.randn_like(latent_image)
                bsz = latent_image.shape[0]

                # 4 Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                # 5 Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latent_image.ndim, dtype=latent_image.dtype)
                noisy_model_input = (1.0 - sigmas) * latent_image + sigmas * noise

                # 6.1 pack noisy_model_input
                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latent_image.shape[0],
                    num_channels_latents=latent_image.shape[1],
                    height=latent_image.shape[2],
                    width=latent_image.shape[3],
                )
                # 6.2 pack Conditions latents
                for i, images_per_condition in enumerate(condition_latents):
                    condition_latents[i] = FluxPipeline._pack_latents(
                        images_per_condition,
                        batch_size=latent_image.shape[0],
                        num_channels_latents=latent_image.shape[1],
                        height=latent_image.shape[2],
                        width=latent_image.shape[3],
                    )

                # 7 handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latent_image.shape[0])
                else:
                    guidance = None
            with accelerator.accumulate(transformer):
                # 8 Predict the noise residual
                model_pred = transformer(
                    model_config={},
                    # Inputs of the condition (new feature)
                    condition_latents=condition_latents,
                    condition_ids=condition_ids,
                    condition_type_ids=None,
                    condition_types = condition_types,
                    # Inputs to the original transformer
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                # flow matching loss
                target = noise - latent_image

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.work_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

