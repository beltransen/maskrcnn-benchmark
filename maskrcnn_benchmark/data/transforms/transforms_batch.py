# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

# random.seed(0) # Reproducibility

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, target):
        if type(images) != list:  # For single-frame datasets
            images = [images]
        for t in self.transforms:
            images, target = t(images, target)

        return images, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, target=None):
        size = self.get_size(images[0].size)
        images = [F.resize(image, size) for image in images]
        if target is None:
            return images
        else:
            target = target.resize(images[0].size)
            return images, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, images, target=None):
        if random.random() < self.prob:
            images = [F.hflip(image) for image in images]

            if target is not None:
                target = target.transpose(0)

        if target is None:
            return images
        else:
            return images, target


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, images, target=None):
        images = [self.color_jitter(image) for image in images]
        if target is None:
            return images
        else:
            return images, target


class ToTensor(object):
    def __call__(self, images, target=None):
        stacked_imgs = F.to_tensor(images[0]).unsqueeze(1)
        for image in images[1:]:
            im = F.to_tensor(image).unsqueeze(1)
            stacked_imgs = torch.cat((stacked_imgs, im), dim=1)
        if target is None:
            return stacked_imgs
        else:
            return stacked_imgs, target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, images, target=None):
        if isinstance(images, list):
            images = ToTensor(images, target)
        for i in range(images.shape[1]):
            if self.to_bgr255:
                images[:, i, :, :] = images[:, i, :, :][[2, 1, 0]] * 255
            images[:, i, :, :] = F.normalize(images[:, i, :, :], mean=self.mean, std=self.std)
        if target is None:
            return images
        return images, target
