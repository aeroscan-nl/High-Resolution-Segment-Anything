import argparse
import time
import cv2
import numpy as np
import torch

from mmengine.config import Config
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = MODELS.build(cfg.model)
    model.load_state_dict(checkpoint)
    model.to("cuda")
    return model


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_model(cfg, args.checkpoint)
    
    image = cv2.imread(args.image)
    image = image.transpose((2, 0, 1))
    image = image[None, :, :, :]
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image).to("cuda")
    # TODO: Might need pre-processing still    
    
    prompt = SegmentorPrompt(
        image=image,
        points=[(130, 600, 'positive')],
        boxes=None
    )
    mask = model(prompt=prompt, mode='prompt')
    cv2.imwrite('mask.png', mask)


if __name__ == '__main__':
    main()
    