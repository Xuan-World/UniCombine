import ipdb
import torch
from typing import Optional, Union, List, Tuple, Any

from torch import Tensor

from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
import numpy as np
import cv2

condition_dict = {
    "depth": 0,
    "canny": 1,
    "subject": 4,
    "coloring": 6,
    "deblurring": 7,
    "fill": 9,
}


class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        no_process: bool = False,
        condition: Union[Image.Image, torch.Tensor] = None,
        condition_ids = None,
        mask=None,
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            if no_process:
                self.condition = raw_img.convert("RGB")
            else:
                self.condition = self.get_condition(condition_type, raw_img)
            self.condition_ids = None
        else:
            self.condition = condition
            self.condition_ids = condition_ids
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"

    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Returns the condition image.
        """
        if condition_type == "depth":
            from transformers import pipeline

            depth_pipe = pipeline(
                task="depth-estimation",
                model="ckpt/depth-anything-small-hf",
                device="cuda",
            )
            source_image = raw_img.convert("RGB")
            condition_img = depth_pipe(source_image)["depth"].convert("RGB")
            return condition_img
        elif condition_type == "canny":
            img = np.array(raw_img)
            edges = cv2.Canny(img, 100, 200)
            edges = Image.fromarray(edges).convert("RGB")
            return edges
        elif condition_type == "subject":
            return raw_img
        elif condition_type == "coloring":
            return raw_img.convert("L").convert("RGB")
        elif condition_type == "deblurring":
            condition_image = (
                raw_img.convert("RGB")
                .filter(ImageFilter.GaussianBlur(10))
                .convert("RGB")
            )
            return condition_image
        elif condition_type == "fill":
            return raw_img.convert("RGB")
        return self.condition

    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]

    def _encode_image(self, pipe: FluxPipeline, cond_img: Image.Image) -> tuple[Any, Any]:
        """
        Encodes an image condition into tokens using the pipeline.
        """
        cond_img = pipe.image_processor.preprocess(cond_img)
        cond_img = cond_img.to("cuda").to(pipe.dtype)
        # cond_img = cond_img.to(pipe.device).to(torch.float16)
        cond_img = pipe.vae.encode(cond_img).latent_dist.sample()
        cond_img = (
            cond_img - pipe.vae.config.shift_factor
        ) * pipe.vae.config.scaling_factor
        cond_tokens = pipe._pack_latents(cond_img, *cond_img.shape)
        cond_ids = pipe._prepare_latent_image_ids(
            cond_img.shape[0],
            cond_img.shape[2]//2,
            cond_img.shape[3]//2,
            pipe.device,
            pipe.dtype,
        )
        if self.condition_type == "subject":
            cond_ids[:, 2] += cond_img.shape[2] //2
        return cond_tokens, cond_ids

    def encode(self, pipe: FluxPipeline) -> tuple[Any, Any, Tensor]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_ids is not None:
            tokens, ids = self.condition, self.condition_ids
        elif self.condition_ids is None and self.condition_type in [
            "depth",
            "canny",
            "subject",
            "coloring",
            "deblurring",
            "fill",
        ]:
            tokens, ids = self._encode_image(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"There are two ways to use it: \n"
                f"(1) Give the condition tensor to the 'self.condition' and the condition_ids to the 'self.condition_ids' manually.\n"
                f"(2) Give the raw_image to the 'self.raw_img' and process the rest operations with a pipeline automatically.\n"
            )
        type_id = torch.ones_like(ids[:, :1]) * self.type_id # the type_id is not used so far.
        return tokens, ids, type_id
