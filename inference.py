import os,sys
import ipdb
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import torch
from src.condition import Condition
from PIL import Image
from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
from src.UniCombinePipeline import UniCombinePipeline
from accelerate.utils import set_seed
import json
import argparse
import cv2
import numpy as np
from datetime import datetime
weight_dtype = torch.bfloat16
device = torch.device("cuda:0")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell/transformer",)
    parser.add_argument("--condition_types", type=str, nargs='+', default=["fill","subject"],)
    parser.add_argument("--denoising_lora",type=str,default="ckpt/Denoising_LoRA/subject_fill_union",)
    parser.add_argument("--denoising_lora_weight",type=float,default=1.0,)
    parser.add_argument("--condition_lora_dir",type=str,default="ckpt/Condition_LoRA",)
    parser.add_argument("--work_dir",type=str,default="output/inference_result",)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--canny",type=str,default=None)
    parser.add_argument("--depth",type=str,default=None)
    parser.add_argument("--fill",type=str,default="examples/window/background.jpg")
    parser.add_argument("--subject",type=str,default="examples/window/subject.jpg")
    parser.add_argument("--json",type=str,default="examples/window/1634_rank0_A decorative fabric topper for windows..json")
    parser.add_argument("--prompt",type=str,default=None)
    parser.add_argument("--num",type=int,default=1)
    parser.add_argument("--version",type=str,default="training-based",choices=["training-based","training-free"])

    args = parser.parse_args()
    args.revision = None
    args.variant = None
    args.json = json.load(open(args.json))
    if args.prompt is None:
        args.prompt = args.json['description']
    args.denoising_lora_name = os.path.basename(os.path.normpath(args.denoising_lora))
    return args




if __name__ == "__main__":
    args = parse_args()
    transformer = UniCombineTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=args.transformer,
    ).to(device = device, dtype=weight_dtype)

    for condition_type in args.condition_types:
        transformer.load_lora_adapter(f"{args.condition_lora_dir}/{condition_type}.safetensors", adapter_name=condition_type)

    pipe = UniCombinePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype = weight_dtype,
        transformer = None
    )
    pipe.transformer = transformer

    if args.version == "training-based":
        pipe.transformer.load_lora_adapter(args.denoising_lora,adapter_name=args.denoising_lora_name, use_safetensors=True)
        pipe.transformer.set_adapters([i for i in args.condition_types] + [args.denoising_lora_name],[1.0,1.0,args.denoising_lora_weight])
    elif args.version == "training-free":
        pipe.transformer.set_adapters([i for i in args.condition_types])

    pipe = pipe.to(device)

    # load conditions
    # "no_process = True" means there is no need to run the canny or depth extraction or any other preparation for the input conditional images.
    # which means the input conditional images can be used directly.
    conditions = []
    for condition_type in args.condition_types:
        if condition_type == "subject":
            conditions.append(Condition("subject", raw_img=Image.open(args.subject), no_process=True))
        elif condition_type == "canny":
            conditions.append(Condition("canny", raw_img=Image.open(args.canny), no_process=True))
        elif condition_type == "depth":
            conditions.append(Condition("depth", raw_img=Image.open(args.depth), no_process=True))
        elif condition_type == "fill":
            conditions.append(Condition("fill", raw_img=Image.open(args.fill), no_process=True))
        else:
            raise ValueError("Only support for subject, canny, depth, fill so far.")

    # load prompt
    prompt = args.prompt

    if args.seed is not None:
        set_seed(args.seed)

    output_dir = os.path.join(args.work_dir, datetime.now().strftime('%y_%m_%d-%H:%M'))
    os.makedirs(output_dir, exist_ok=True)

    # generate
    for i in range(args.num):
        result_img = pipe(
            prompt=prompt,
            conditions=conditions,
            height=512,
            width=512,
            num_inference_steps=8,
            max_sequence_length=512,
            model_config = {},
        ).images[0]

        concat_image = Image.new("RGB", (512 + len(args.condition_types) * 512, 512))
        for j, cond_type in enumerate(args.condition_types):
            cond_image = conditions[j].condition
            if cond_type == "fill":
                cond_image = cv2.rectangle(np.array(cond_image), args.json['bbox'][:2], args.json['bbox'][2:], color=(128, 128, 128),thickness=-1)
                cond_image = Image.fromarray(cv2.rectangle(cond_image, args.json['bbox'][:2], args.json['bbox'][2:], color=(255, 215, 0), thickness=2))
            concat_image.paste(cond_image, (j * 512, 0))
        concat_image.paste(result_img, (j * 512 + 512, 0))
        concat_image.save(os.path.join(output_dir, f"{i}_result.jpg"))
        print(f"Done. Output saved at {output_dir}/{i}_result.jpg")