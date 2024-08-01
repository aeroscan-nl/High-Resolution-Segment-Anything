from models.segmentors.promptseg import SegmentorPrompt
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmseg.registry import MODELS
from models.segmentors.promptseg import SegmentorPrompt


class HrSamPredictor:
    def __init__(self, checkpoint_path: str, cfg_path: str, device: str = "cuda"):
        self.device = device
        
        cfg = Config.fromfile(cfg_path)
        self.model, self.preprocessor = self.__build_model(cfg, checkpoint_path)
        self.image_embeddings = None
        self.logits = None
        self.image = None
        
    def set_image(self, image: np.ndarray):
        """
        Sets the image for segmentation.

        Args:
            image (np.ndarray): The input image for segmentation. It should have shape (H, W, 3).
        """
        if image.shape[-1] != 3:
            raise ValueError("Image should have shape (H, W, 3)")
        
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        data = self.preprocessor({'inputs': [image], 'data_samples': []})
        image = data['inputs'][0]
        image = image[None, :, :, :].to(self.device)
        
        self.image = image
        self.image_embeddings = self.model.encode_image(image)
        self.logits = None
    
    def predict(
            self,
            point_coords: Optional[List[Tuple[int, int, int]]] = None,
            box: Optional[np.ndarray] = None,
            mask_input: Optional[np.ndarray] = None,
            multimask_output: bool = False,
            return_logits: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Predicts the mask for the given image using the High-Resolution Segment Anything model.

            Args:
                point_coords (Optional[List[Tuple[int, int, int]]]): A list of tuples representing the coordinates of the points.
                    Each tuple should contain three values: x-coordinate, y-coordinate, and a binary value (0 or 1) indicating
                    whether the point is positive or negative.
                box (Optional[np.ndarray]): Not yet implemented.
                mask_input (Optional[np.ndarray]): Not yet implemented.
                multimask_output (bool): Not yet implemented.
                return_logits (bool): Not yet implemented.

            Returns:
                (np.ndarray): The output masks in CxHxW format, where C is the
                    number of masks, and (H, W) is the original image size.
                (np.ndarray): An array of length C containing the model's
                    predictions for the quality of each mask.
                (np.ndarray): An array of shape CxHxW, where C is the number
                    of masks and H=W=256. These low resolution logits can be passed to
                    a subsequent iteration as mask input.
            """
            
            if box or mask_input or multimask_output or return_logits:
                raise NotImplementedError("Not yet implemented")
            
            if self.image_embeddings is None or self.image is None:
                raise ValueError("Use set_image() to set the image before calling predict()")
            
            prompt = SegmentorPrompt(
                image=self.image,
                image_embeddings=self.image_embeddings,
                points=point_coords,
                boxes=None,
                previous_logits=self.logits
            )
            
            masks, ious, logits = self.model(prompt=prompt, mode='prompt')
            self.logits = logits
            return masks, ious, logits
    
    def __build_model(self, cfg, checkpoint_path) -> Tuple[torch.nn.Module, torch.nn.Module]:
        preprocessor = MODELS.build(cfg.data_preprocessor)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model = MODELS.build(cfg.model)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        return model, preprocessor
    
    
if __name__ == '__main__':
    predictor = HrSamPredictor(
        checkpoint_path='work_dirs/hrsam/coco_lvis/simdist_hrsam_plusplus_colaug_1024x1024_bs1_160k/iter_160000.pth',
        cfg_path='configs/inference/hrsam_plusplus_simaug_1024x1024.py')
    image = cv2.imread('data/testing/ORT2.jpg')
    predictor.set_image(image)
    mask = predictor.predict(point_coords=[(100, 100, 1), (200, 200, 1)])
    