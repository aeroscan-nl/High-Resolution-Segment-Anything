import argparse
import cv2
import torch

from mmengine.config import Config
from mmengine.runner import set_random_seed
from mmseg.registry import MODELS

from models.segmentors.promptseg import SegmentorPrompt


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with HR-SAM')
    parser.add_argument('config', help='path to model config file')
    parser.add_argument('checkpoint', help='path to checkpoint file')
    parser.add_argument('image', help='path to image file')
    
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
    image = data['inputs'][0]
    image = image[None, :, :, :].to("cuda")
    return image


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    set_random_seed(cfg.randomness.seed)
    
    model, preprocessor = build_model(cfg, args.checkpoint)
    image = load_image(args.image, preprocessor)
    points = [
        (99, 118, 'positive'),
        (309, 276, 'negative'),
        (156, 323, 'negative'),
        (440, 189, 'negative'),
        (361, 139, 'positive'),
        (371, 115, 'negative'),
        (366, 121, 'positive'),
        (357, 114, 'positive'),
        (358, 112, 'positive'),
        (67, 297, 'negative'),
        (360, 115, 'negative'),
        (371, 113, 'negative'),
        (365, 124, 'positive'),
        (365, 118, 'positive'),
        (369, 121, 'positive'),
        (371, 117, 'negative'),
        (368, 124, 'positive'),
        (372, 114, 'negative'),
        (374, 89, 'positive'),
        (364, 125, 'positive')
    ]
    
    prev_logits = None
    for i in range(len(points)):
        prompt = SegmentorPrompt(
            image=image,
            points=points[:i + 1],
            boxes=None,
            logits=prev_logits
        )
        mask, prev_logits = model(prompt=prompt, mode='prompt')
        cv2.imwrite(f'mask{i}.png', mask)


if __name__ == '__main__':
    main()
    