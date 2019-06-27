#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Demo [Single or Multi-frame (NLNets)]")
    parser.add_argument(
        "--config-file",
        default="../configs/e2e_faster_rcnn_R_50_FPN_1x_KITTI_nonlocal_pret.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--image",
        default="../demo/demo_e2e_mask_rcnn_R_50_FPN_1x.png",
        metavar="FILE",
        help="path to image file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=375,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    img = cv2.imread(args.image)
    imgs = [img]

    start_time = time.time()
    if '_nl' in cfg.DATASETS.TRAIN[0]:
        print("Hey man, you're using NLNets")
        prev_dir = args.image
        prev_dir = prev_dir.replace("image", "prev")
        # Load previous frames into variables
        for prev_id in range(1, 4):
            complete_name_prev_img = prev_dir[:-4] + '_{0:02d}'.format(prev_id) + '.png'
            if os.path.exists(complete_name_prev_img):
                img_prev = cv2.imread(complete_name_prev_img)
                imgs.append(img_prev)
            else:
                break
        composite = coco_demo.run_on_opencv_sequence(imgs)
    else:
        composite = coco_demo.run_on_opencv_image(imgs)

    print("Time: {:.2f} s / img".format(time.time() - start_time))
    cv2.imshow("Detections", composite)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()
