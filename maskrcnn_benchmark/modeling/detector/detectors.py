# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN, GeneralizedRCNN3D


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedRCNN3D": GeneralizedRCNN3D}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
