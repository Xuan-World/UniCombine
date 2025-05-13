import pandas as pd
import json
import ast
import csv
from torch.utils.data import Dataset
from PIL import Image
import ipdb
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from src.condition import Condition
from diffusers.image_processor import VaeImageProcessor
import torch
class StyleDataset(Dataset):
    def __init__(self, csv_path, root_dir='', transform=None,args = None):
        self.datalist = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 只读取第一条数据
            for row in reader:
                # 解析SearchResult字段为字典
                style_img = list(json.loads(row['SearchResult:DICT']).keys())
                style_name=row['ShortStyleName']
                target_file=row['Target:FILE']
                source_file=row['Source:FILE']
                self.datalist.append({"style":style_name,"ref":style_img,"tgt":target_file,"src":source_file})
        self.root_dir = root_dir
        self.transform = transform
        self.image_processor = VaeImageProcessor(vae_scale_factor=args.vae_scale_factor * 2 ,do_resize=True,do_convert_rgb=True)
        self.resolution = args.resolution
    def __len__(self):
        return len(self.datalist)
    def load_image(self,pth):
        return Image.open(pth).convert('RGB')
    def __getitem__(self, idx):
        item = self.datalist[idx]
        target = self.load_image(os.path.join(self.root_dir, item['tgt']))
        source = self.load_image(os.path.join(self.root_dir, item['src']))
        style_images = [self.load_image(os.path.join(self.root_dir, style_path)) for style_path in item['ref']]
        description = item['style']
        if self.transform:
            target = self.transform(target)
            source = self.transform(source)
            style_images = [self.transform(img) for img in style_images]
        pixel_values = self.image_processor.preprocess(target,width=self.resolution,height=self.resolution).squeeze(0)
        condition_latents = torch.stack([self.image_processor.preprocess(random.choice(style_images),width=self.resolution,height=self.resolution).squeeze(0)])
        return {"pixel_values":pixel_values,"condition_latents":condition_latents,"description":description, "condition_types":['style']}

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    condition_latents = torch.stack([example["condition_latents"] for example in examples])
    condition_latents = condition_latents.to(memory_format=torch.contiguous_format).float()
    condition_types= [example["condition_types"] for example in examples]
    descriptions = [example["description"] for example in examples]
    return {"pixel_values": pixel_values, "condition_latents": condition_latents,
            "condition_types":condition_types,"descriptions": descriptions}

