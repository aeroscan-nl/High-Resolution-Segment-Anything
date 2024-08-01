import numpy as np
import torch
from mmseg.registry import MODELS

from .base import BaseClickSegmentor

class SegmentorPrompt:
    def __init__(self, image, image_embeddings, points=None, boxes=None, previous_logits=None):
        self.image = image
        self.image_embeddings = image_embeddings
        self.points = points
        self.boxes = boxes
        self.previous_logits = previous_logits


@MODELS.register_module()
class PromptSegmentor(BaseClickSegmentor):
    """
    Segmentor that performs standard forward passes using user prompts.
    """
    @torch.no_grad()
    def prompted_inference(self, prompt: SegmentorPrompt):
        if prompt.image_embeddings is None:
            image_embeds = self.encode_image(prompt.image)
            prompt.image_embeddings = image_embeds
        
        logits, ious = self._decode_prompt(prompt)
       
        high_res_masks = self.interpolate(logits, self.test_cfg.target_size)
        high_res_masks = self.crop_and_resize_to_original_size(
            high_res_masks, prompt.image.shape[-2:], self.test_cfg.target_size)
        high_res_masks = high_res_masks > 0.0
        high_res_masks = high_res_masks.squeeze().detach().cpu().numpy()
        
        logits = logits.detach().cpu().numpy()
        ious = ious.detach().cpu().numpy()
        
        return high_res_masks, ious, logits

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        self.eval()
        
        resized_padded_inputs = self.resize_and_pad_to_target_size(image, self.test_cfg.target_size)
        image_embeds = self.backbone(resized_padded_inputs)
        
        return image_embeds
    
    def _decode_prompt(self, prompt: SegmentorPrompt):
        self.eval()
        
        # x and y coords need to be swapped...idk why yet
        points = [(y, x, label) for x, y, label in prompt.points]
        points = self.resize_coord_to_target_size(
            points, prompt.image.shape[-2:], self.test_cfg.target_size, prompt.image.device)
        prompt_embeds = self.neck(
            image_embeds=prompt.image_embeddings,
            points=points,
            boxes=None,
            prev_logits=prompt.previous_logits)
        
        # Decode image + prompt embeddings and post-process mask
        logits, ious = self.decode_head(prompt_embeds, mode='multiple masks with ious')
        
        return logits, ious
    