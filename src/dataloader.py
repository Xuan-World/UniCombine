from accelerate.logging import get_logger
import torch
import io
logger = get_logger(__name__)
from PIL import Image
from .condition import Condition
from diffusers.image_processor import VaeImageProcessor
from datasets import load_dataset, concatenate_datasets
def get_dataset(args):
    dataset = []
    assert isinstance(args.dataset_name,list),"dataset dir should be a list"
    if args.dataset_name is not None:
        for name in args.dataset_name:
            # Downloading and loading a dataset from the hub.
            dataset.append(load_dataset(name,cache_dir=args.cache_dir,split='train'))
    dataset = concatenate_datasets(dataset)
    return dataset

def prepare_dataset(dataset, vae_scale_factor, accelerator, args):
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2 ,do_resize=True,do_convert_rgb=True)

    def preprocess_conditions(conditions):
        conditioning_tensors = []
        conditions_types = []
        for cond in conditions:
            conditioning_tensors.append(image_processor.preprocess(cond.condition,width=args.resolution,height=args.resolution).squeeze(0))
            conditions_types.append(cond.condition_type)
        return torch.stack(conditioning_tensors,dim=0),conditions_types
    def preprocess(examples):
        # images = [image_transforms(image) for image in images]
        pixel_values =[]
        condition_latents=[]
        condition_types=[]
        bboxes = []
        for image,bbox,canny,depth in zip(examples[args.image_column],examples[args.bbox_column],examples[args.canny_column],examples[args.depth_column]):
            image = image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB")
            width, height = image.size
            # 检查宽度是否为偶数，以便可以均匀分割
            if width % 2 != 0:
                raise ValueError("Image width must be even to split into two equal parts.")
            # 分割图像
            left_image = image.crop((0, 0, width // 2, height))  # 左半部分
            right_image = image.crop((width // 2, 0, width, height))  # 右半部分
            # load mask image
            image_width,image_height = image.size
            bbox_pixel = [
                bbox[0] * image_width,
                bbox[1] * image_height,
                bbox[2] * image_width,
                bbox[3] * image_height
            ]
            left = bbox_pixel[0] - bbox_pixel[2] / 2
            top = bbox_pixel[1] - bbox_pixel[3] / 2
            right = bbox_pixel[0] + bbox_pixel[2] / 2
            bottom = bbox_pixel[1] + bbox_pixel[3] / 2
            masked_left_image = left_image.copy()
            masked_left_image.paste((0, 0, 0), (int(left), int(top), int(right), int(bottom)))
            bboxes.append([int(left*args.resolution/(width // 2)), int(top*args.resolution/height), int(right*args.resolution/(width // 2)), int(bottom*args.resolution/height)])
            # 应用转换,将分割后的图像添加到列表中
            pixel_values.append(image_processor.preprocess(left_image,width=args.resolution,height=args.resolution).squeeze(0))
            conditions = []
            for condition_type in args.condition_types:
                if condition_type == "subject":
                    conditions.append(Condition("subject", condition = right_image))
                elif condition_type == "canny":
                    conditions.append(Condition("canny", condition = Image.open(io.BytesIO(canny['bytes']))))
                elif condition_type == "depth":
                    conditions.append(Condition("depth", condition = Image.open(io.BytesIO(depth['bytes']))))
                elif condition_type == "fill":
                    conditions.append(Condition("fill", condition = masked_left_image))
                else:
                    raise ValueError("Only support for subject, canny, depth, fill")
            cond_tensors, cond_types = preprocess_conditions(conditions)
            condition_latents.append(cond_tensors)
            condition_types.append(cond_types)
        examples["pixel_values"] = pixel_values
        examples["condition_latents"] = condition_latents
        examples["condition_types"] = condition_types
        examples["bbox"]=bboxes
        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess)

    return dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    condition_latents = torch.stack([example["condition_latents"] for example in examples])
    condition_latents = condition_latents.to(memory_format=torch.contiguous_format).float()
    bboxes= [example["bbox"] for example in examples]
    condition_types= [example["condition_types"] for example in examples]
    descriptions = [example["description"]["description_0"] for example in examples]
    items = [example["description"]["item"] for example in examples]
    return {"pixel_values": pixel_values, "condition_latents": condition_latents,
            "condition_types":condition_types,"descriptions": descriptions, "bboxes": bboxes,"items":items}

