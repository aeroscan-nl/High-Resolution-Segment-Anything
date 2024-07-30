import argparse
import json
import cv2
import torch

from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.registry import MODELS

from models.segmentors.promptseg import SegmentorPrompt
from tools.utils.sam import get_sam_polygon
from tools.utils.visualization import draw_sam_polygon_on_image


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with HR-SAM')
    parser.add_argument('config', help='path to model config file')
    parser.add_argument('checkpoint', help='path to checkpoint file')
    parser.add_argument('image', help='path to image file')
    parser.add_argument('points', help='sam point prompt')
    
    args = parser.parse_args()
    return args


def build_model(cfg, checkpoint_path):
    preprocessor = MODELS.build(cfg.data_preprocessor)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = MODELS.build(cfg.model)
    model.load_state_dict(checkpoint)
    model.to("cuda")
    return model, preprocessor


def load_image(img_path, preprocessor):
    image = cv2.imread(img_path)
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image).to("cuda")
    data = preprocessor({'inputs': [image], 'data_samples': []})
    processed_image = data['inputs'][0]
    processed_image = processed_image[None, :, :, :].to("cuda")
    return processed_image, image


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.randomness.seed)
    
    model, preprocessor = build_model(cfg, args.checkpoint)
    image, original_image = load_image(args.image, preprocessor)
    
    if args.points:
        all_points = json.loads(args.points)
    else:
        raise Exception('No points provided')
    
    prev_logits = None
    for i in range(len(all_points)):
        points = all_points[:i + 1]
        prompt = SegmentorPrompt(
            image=image,
            points=points,
            boxes=None,
            logits=prev_logits
        )
        mask, prev_logits = model(prompt=prompt, mode='prompt')
        
        cv2.imwrite(f'work_dirs/testing/mask{i}.png', mask)
        
        polygon = get_sam_polygon(mask, points)
        if len(polygon) > 100: continue
        polygon_image = original_image.squeeze().cpu().numpy().transpose((1, 2, 0))
        polygon_image = draw_sam_polygon_on_image(polygon_image, polygon, fill_color=(255, 0, 0))
        
        cv2.imwrite(f'work_dirs/testing/mask_overlay{i}.png', polygon_image)


if __name__ == '__main__':
    main()
    