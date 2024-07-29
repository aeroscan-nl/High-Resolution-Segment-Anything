import torch
from mmseg.registry import MODELS

from .base import BaseClickSegmentor

class SegmentorPrompt:
    def __init__(self, image, points=None, boxes=None, logits=None):
        self.image = image
        self.points = points
        self.boxes = boxes
        self.logits = logits


@MODELS.register_module()
class PromptSegmentor(BaseClickSegmentor):
    """
    Segmentor that performs standard forward passes using user prompts.
    """

    @torch.no_grad()
    def prompted_inference(self, prompt: SegmentorPrompt):
        cfg = self.test_cfg
        self.eval()

        # Encode image 
        resized_padded_inputs = self.resize_and_pad_to_target_size(prompt.image, cfg.target_size)
        image_embeds = self.backbone(resized_padded_inputs)
        
        # Encode image embeddings with prompt
        points = self.resize_coord_to_target_size(
            prompt.points, prompt.image.shape[-2:], cfg.target_size, prompt.image.device)
        prompt_embeds = self.neck(
            image_embeds=image_embeds,
            points=points,
            boxes=prompt.boxes,
            prev_logits=prompt.logits)
        
        # Decode image + prompt embeddings and post-process mask
        logits = self.decode_head(prompt_embeds, mode='single mask')
        mask = self.interpolate(logits, resized_padded_inputs.shape[-2:])
        mask = self.crop_and_resize_to_original_size(
            mask, prompt.image.shape[-2:], cfg.target_size)
        mask = mask > 0.0
        mask = (mask.squeeze().clamp(0, 255) * 255).to(torch.uint8)
        mask = mask.squeeze().detach().cpu().numpy()
        
        return mask, logits
