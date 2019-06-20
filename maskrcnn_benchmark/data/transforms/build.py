# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T
from . import transforms_batch as TB


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    if cfg.NON_LOCAL.ENABLED:  # Input is a 4D Tensor (Image sequences)
        transformer = TB  # Apply the same transformations for a group of images stored in a list
    else:
        transformer = T  # Common single image transformations
    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = transformer.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = transformer.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = transformer.Compose(
        [
            color_jitter,
            transformer.Resize(min_size, max_size),
            transformer.RandomHorizontalFlip(flip_prob),
            transformer.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
