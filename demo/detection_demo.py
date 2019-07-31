#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import os
import cv2
import numpy as np
from termcolor import colored

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time

refPt = []
image = None
image_composition = None
original_composition = None
net_input_size = None
my_cfg = None
att_maps = None
active_layer = None
downsample_factor = dict()


def show_non_local_connections(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, image, image_composition, original_composition, net_input_size, my_cfg, att_maps, active_layer
    assert image is not None

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed

    downsample_factor[active_layer] = 32
    if event == cv2.EVENT_LBUTTONDBLCLK:
        image_composition = np.copy(original_composition)  # Fetch unmodified composition
        refPt = [x, y]

        # Get point coords at network input's scale
        input_min = my_cfg.INPUT.MIN_SIZE_TEST
        image_size = image.shape[:2]
        image_min = min(image_size)
        scale_factor = input_min/image_min
        # print('Scale factor', scale_factor)
        new_size = scale_factor*np.asarray(image_size)
        new_size = tuple(new_size.astype(np.int))

        # Get point at target layer scale
        targetPt = np.asarray(refPt) * scale_factor
        targetPt = refPt
        # print('{} -> {}'.format(refPt, targetPt))

        # Fetch attention map for that pixel
        attPt = np.asarray(targetPt) // downsample_factor[active_layer]
        attPt = attPt.astype(np.int)
        # print('[{}] {} -> {}'.format(active_layer, targetPt, attPt))
        pixel_att = att_maps[active_layer][0, attPt[1], attPt[0], :, :]
        # print('Size ', pixel_att.shape)

        # Compute attention map window at input size scale
        pixel_att_resized = []
        for att_window in pixel_att:
            att_window = np.reshape(att_window, (my_cfg.NON_LOCAL_CTX.PATCH_SIZE, my_cfg.NON_LOCAL_CTX.PATCH_SIZE))
            pixel_att_resized.append(cv2.resize(att_window, tuple(np.asarray(att_window.shape)*downsample_factor[active_layer]), interpolation=cv2.INTER_NEAREST))

        img_height = image_composition.shape[0] // (len(pixel_att_resized) + 1)
        img_width = image_composition.shape[1]

        # Compute patch and receptive field regions
        patch_width, patch_height = pixel_att_resized[0].shape
        min_col = max(0, attPt[0] * downsample_factor[active_layer] - patch_width // 2)
        max_col = min(img_width, attPt[0] * downsample_factor[active_layer] + patch_width // 2)
        min_row = max(0, attPt[1] * downsample_factor[active_layer] - patch_height // 2)
        max_row = min(img_height, attPt[1] * downsample_factor[active_layer] + patch_height // 2)
        valid_width = max_col - min_col
        valid_height = max_row - min_row

        # Default indexes contains the whole patch
        min_col_idx = 0
        max_col_idx = patch_width
        min_row_idx = 0
        max_row_idx = patch_height

        # Adjust them if needed
        if min_col == 0:
            min_col_idx = -valid_width
        elif max_col == img_width:
            max_col_idx = valid_width

        if min_row == 0:
            min_row_idx = -valid_height
        elif max_row == img_height:
            max_row_idx = valid_height

        # For every preceding frame, represent tiles proportional to their attention score
        hsv_composition = cv2.cvtColor(image_composition, cv2.COLOR_BGR2HSV)
        for i in range(len(pixel_att_resized)):
            # frame = pixel_att_resized[i] * 255
            # cv2.imshow('att_resize', frame.astype(np.uint8))

            # print('Copying Patch [Rows: ({}, {}) Cols: ({}, {})] into Image [Rows: ({}, {}) Cols: ({}, {})]'
            #       .format(min_row_idx, max_row_idx, min_col_idx, max_col_idx, min_row, max_row, min_col, max_col))
            row_offset = (i+1) * img_height
            hsv_composition[row_offset+min_row:row_offset+max_row, min_col:max_col, 2] = \
                pixel_att_resized[i][min_row_idx:max_row_idx, min_col_idx:max_col_idx] * hsv_composition[row_offset+min_row:row_offset+max_row, min_col:max_col, 2]

            cv2.rectangle(hsv_composition,
                          (max(0, attPt[0]*downsample_factor[active_layer] - pixel_att_resized[i].shape[0]//2),
                           max(img_height * (i + 1), img_height * (i + 1) + attPt[1]*downsample_factor[active_layer] - pixel_att_resized[i].shape[1]//2)),
                          (min(img_width, attPt[0]*downsample_factor[active_layer] + pixel_att_resized[i].shape[0]//2),
                           min(img_height * (i + 2), img_height * (i + 1) + attPt[1]*downsample_factor[active_layer] + pixel_att_resized[i].shape[1]//2)),
                          (0, 255, 255),  # hsv red
                          5
                          )
        image_composition = cv2.cvtColor(hsv_composition, cv2.COLOR_HSV2BGR)

        # Scale down the images to original size
        cv2.circle(image_composition, (refPt[0], refPt[1]), 5, (0, 0, 255), -1)

        # Draw receptive field in current frame
        cv2.rectangle(image_composition,
                      (attPt[0]*downsample_factor[active_layer],
                       attPt[1]*downsample_factor[active_layer]),
                      (attPt[0]*downsample_factor[active_layer]+downsample_factor[active_layer],
                       attPt[1]*downsample_factor[active_layer]+downsample_factor[active_layer]), (0, 255, 0), 5)
        #


def main():
    global refPt, image, image_composition, original_composition, net_input_size, my_cfg, att_maps, active_layer
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Demo [Single or Multi-frame (NLNets)]")
    parser.add_argument(
        "--config-file",
        default="../configs/kitti2d/e2e_faster_rcnn_R_50_FPN_1x_Alabama.yaml",
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

    my_cfg = cfg.clone()
    opts = ["NON_LOCAL_CTX.RETURN_ATTENTION", True]
    my_cfg.merge_from_list(opts)  # If NL, make sure attention is shown

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        my_cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    img = cv2.imread(args.image)
    imgs = [img]

    start_time = time.time()
    if '_nl' in my_cfg.DATASETS.TRAIN[0]:
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
        image = coco_demo.run_on_opencv_sequence(imgs)
        if my_cfg.NON_LOCAL_CTX.RETURN_ATTENTION:
            att_maps = image[1]
            for key in att_maps:
                # print('{}: {}'.format(key, att_maps[key].shape))
                att_maps[key] = att_maps[key].cpu().numpy()
                active_layer = key  # Init to last layer (random)
                downsample_factor[key] = np.ceil(my_cfg.INPUT.MIN_SIZE_TEST/att_maps[key].shape[1])  # Compute factor
                net_input_size = (att_maps[key].shape[1] * downsample_factor[key],
                                  att_maps[key].shape[2] * downsample_factor[key])
            image = image[0]
    else:
        image = coco_demo.run_on_opencv_image(imgs)

    print("Time: {:.2f} s / img".format(time.time() - start_time))

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

    if my_cfg.NON_LOCAL_CTX.ENABLED:
        cv2.setMouseCallback("Detections", show_non_local_connections)

    input_min = my_cfg.INPUT.MIN_SIZE_TEST
    image_size = image.shape[:2]
    image_min = min(image_size)
    scale_factor = input_min / image_min
    # print('Scale factor', scale_factor)
    new_size = scale_factor * np.asarray(image_size)
    new_size = new_size.astype(np.int)
    # print('New size ', new_size)

    image_composition = np.zeros((new_size[0]*len(imgs), new_size[1], image.shape[2]), dtype=np.uint8)
    for i in range(len(imgs)):
        image_composition[i*new_size[0]:(i+1)*new_size[0]:, :, :] = cv2.resize(imgs[i], (new_size[1], new_size[0]),
                                                                               interpolation=cv2.INTER_CUBIC)

    original_composition = np.copy(image_composition)  # Backup

    while True:
        # display the image and wait for a keypress
        cv2.imshow("Detections", image_composition)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif 47 < key < 58:  # If it's a number, change attention layer

            if my_cfg.NON_LOCAL_CTX.ENABLED:
                key = key - 48  # 48 is number 0
                aux = 'layer{}'.format(key)
                if aux in att_maps.keys():  # Only update if layer exists
                    active_layer = aux
                    print(colored('Switched to {}'.format(active_layer), 'green'))
                else:
                    print(colored('[ERROR] Requested attention layer [{}] does not exist'.format(aux), 'red'))

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()

'''
body out torch.Size([1, 256, 128, 416]) 4x
body out torch.Size([1, 512, 64, 208]) 8x
body out torch.Size([1, 1024, 32, 104]) 16x
body out torch.Size([1, 2048, 16, 52]) 32x

'''