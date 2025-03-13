import sys,os
import numpy as np
from PIL import Image
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import argparse
import logging
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import diffusers
from diffusers import FluxPipeline
import json
from diffusers.image_processor import VaeImageProcessor
from src.condition import Condition
from diffusers.utils import check_min_version, is_wandb_available
from src.dataloader import get_dataset,prepare_dataset,collate_fn
from datetime import datetime
if is_wandb_available():
    pass
import cv2
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
from src.UniCombinePipeline import UniCombinePipeline


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="testing script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell/transformer",)
    parser.add_argument(
        "--dataset_name",type=str,
        default=[
            "dataset/split_SubjectSpatial200K/test",
            "dataset/split_SubjectSpatial200K/Collection3/test"
        ],
    )
    parser.add_argument("--image_column", type=str, default="image", )
    parser.add_argument("--bbox_column", type=str, default="bbox", )
    parser.add_argument("--canny_column", type=str, default="canny", )
    parser.add_argument("--depth_column", type=str, default="depth", )
    parser.add_argument("--condition_types", type=str, nargs='+', default=["fill", "subject"], )
    parser.add_argument("--denoising_lora",type=str,default="ckpt/Denoising_LoRA/subject_fill_union",)
    parser.add_argument("--condition_lora_dir",type=str,default="ckpt/Condition_LoRA",)
    parser.add_argument("--max_sequence_length",type=int,default=512,help="Maximum sequence length to use with with the T5 text encoder")
    parser.add_argument("--work_dir",type=str,default="output/test_result")
    parser.add_argument("--cache_dir",type=str,default="cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers",type=int,default=0,)
    parser.add_argument("--mixed_precision",type=str,default="bf16",choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed running: local_rank")


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    args.revision = None
    args.variant = None
    args.denoising_lora_name = os.path.basename(os.path.normpath(args.denoising_lora))
    return args


def main(args):
    # 1. set the accelerator and logger
    accelerator_project_config = ProjectConfiguration(project_dir=args.work_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 2. set seed
    if args.seed is not None:
        set_seed(args.seed)

    # 3. create the working directory
    if accelerator.is_main_process:
        if args.work_dir is not None:
            os.makedirs(args.work_dir, exist_ok=True)

    # 4. precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 5. Load pretrained single conditional LoRA modules onto the FLUX transformer
    transformer = UniCombineTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.transformer,
        revision=args.revision,
        variant=args.variant
    ).to(accelerator.device, dtype=weight_dtype)
    lora_names = args.condition_types
    for condition_type in lora_names:
        transformer.load_lora_adapter(f"{args.condition_lora_dir}/{condition_type}.safetensors", adapter_name=condition_type)
    logger.info("You are working on the following condition types: {}".format(lora_names))

    # 6. get the inference pipeline.
    pipe = UniCombinePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=None,
    ).to(accelerator.device, dtype=weight_dtype)
    pipe.transformer = transformer

    # 7. get the VAE image processor to do the pre-process and post-process for images.
    # (vae_scale_factor is the scale of downsample.)
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2 ,do_resize=True)

    # 8. get the dataset
    dataset = get_dataset(args)
    print("len:",len(dataset))
    dataset = prepare_dataset(dataset, vae_scale_factor, accelerator, args)

    # 9. set the seed
    if args.seed is not None:
        set_seed(args.seed)

    # 10. get the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # 10. accelerator start
    initial_global_step = 0
    pipe, dataloader = accelerator.prepare(
        pipe, dataloader
    )

    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Transformer Class = {transformer.__class__.__name__}")
    logger.info(f"  Num of GPU processes = {accelerator.num_processes}")


    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    output_dir = os.path.join(args.work_dir, f"{datetime.now().strftime("%y:%m:%d-%H:%M")}")
    logger.info(f"output dir: {output_dir}")
    os.makedirs(os.path.join(output_dir, "info"), exist_ok=True)

    # 11. start testing!
    for S, batch in enumerate(dataloader):
        prompts = batch["descriptions"]
        # 12.1 Get Conditions input tensors -> "condition_latents"
        # 12.2 Get Conditions positional id list. -> "condition_ids"
        # 12.3 Get Conditions types string list. -> "condition_types"
        # (bs, cond_num, c, h, w) -> [cond_num, (bs, c, h ,w)]
        condition_latents = list(torch.unbind(batch["condition_latents"], dim=1))
        # [cond_num, (len ,3) ]
        condition_ids = []
        # [cond_num]
        condition_types = batch["condition_types"][0]
        for i,images_per_condition in enumerate(condition_latents):
            # i means No.i. Conditional Branch
            # images_per_condition = (bs, c, h ,w)
            images_per_condition = encode_images(pixels=images_per_condition,vae=pipe.vae,weight_dtype=weight_dtype)
            condition_latents[i] = FluxPipeline._pack_latents(
                images_per_condition,
                batch_size=images_per_condition.shape[0],
                num_channels_latents=images_per_condition.shape[1],
                height=images_per_condition.shape[2],
                width=images_per_condition.shape[3],
            )
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

        # 13 prepare the input conditions=[Condition1, Condition2, ...] for all the conditional branches
        conditions = []
        for i, condition_type in enumerate(condition_types):
            conditions.append(Condition(condition_type,condition=condition_latents[i],condition_ids=condition_ids[i]))

        # 14.1 inference of training-based UniCombine
        pipe.transformer.load_lora_adapter(args.denoising_lora,adapter_name=args.denoising_lora_name ,use_safetensors=True)
        pipe.transformer.set_adapters([i for i in lora_names] + [args.denoising_lora_name ])
        if args.seed is not None:
            set_seed(args.seed)
        result_img_list = pipe(
            prompt=prompts,
            conditions=conditions,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=8,
            max_sequence_length=512,
            model_config = {
            },
            accelerator=accelerator
        ).images
        pipe.transformer.delete_adapters(args.denoising_lora_name)

        # 14.2 inference of training-free UniCombine
        pipe.transformer.set_adapters([i for i in lora_names])
        if args.seed is not None:
            set_seed(args.seed)
        origin_result_img_list = pipe(
            prompt=prompts,
            conditions=conditions,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=8,
            max_sequence_length=512,
            model_config = {
            },
            accelerator = accelerator
        ).images

        # 15. save the output to the output_dir = "work_dir/datetime"
        for i,(result_img,origin_result_img) in enumerate(zip(result_img_list,origin_result_img_list)):
            target_img = image_processor.postprocess(batch["pixel_values"][i].unsqueeze(0),output_type="pil")[0]
            cond_images = image_processor.postprocess(batch["condition_latents"][i],output_type="pil")
            concat_image = Image.new("RGB", (1536+len(cond_images)*512, 512))
            for j,(cond_image,cond_type) in enumerate(zip(cond_images,condition_types)):
                if cond_type == "fill":
                    cond_image = cv2.rectangle(np.array(cond_image), tuple(batch['bboxes'][i][:2]),tuple(batch['bboxes'][i][2:]), color=(128, 128, 128), thickness=-1)
                    cond_image = Image.fromarray(cv2.rectangle(cond_image, tuple(batch['bboxes'][i][:2]), tuple(batch['bboxes'][i][2:]),color=(255, 215, 0), thickness=2))
                concat_image.paste(cond_image,(j*512,0))
            concat_image.paste(result_img,(j*512+512,0))
            concat_image.paste(origin_result_img,(j*512+1024,0))
            concat_image.paste(target_img,(j*512+1536,0))

            concat_image.save(os.path.join(output_dir,f"{S*args.batch_size+i}_{batch['items'][i]}.jpg"))

            with open(os.path.join(output_dir,"info",f"{S*args.batch_size+i}_rank{accelerator.local_process_index}_{batch['items'][i]}.json"), "w", encoding="utf-8") as file:
                meta_data = {
                    "description": prompts[i],
                    "bbox": batch["bboxes"][i]
                }
                json.dump(meta_data,file, ensure_ascii=False, indent=4)

        progress_bar.update(1)

if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        main(args)

