# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
import numpy as np
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None, kitti_output=False
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions_multi = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions_multi, os.path.join(output_folder, "predictions.pth"))

    if kitti_output:
        CATEGORIES = [
            "__background",
            "Car",
            "Pedestrian",
            "Cyclist",
        ]

        # for image_id, prediction in enumerate(predictions):
        for image_id in range(len(predictions.keys())):
            original_id = dataset.id_to_img_map[image_id]
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = predictions[image_id]
            prediction = prediction.resize((image_width, image_height))
            file_name = os.path.join(output_folder, 'det_results', '{0:06d}.txt'.format(original_id))
            with open(file_name, 'w') as output_file:
                scores = predictions[image_id].get_field("scores").tolist()
                labels = predictions[image_id].get_field("labels").tolist()
                labels = [CATEGORIES[i] for i in labels]
                boxes = prediction.bbox
                for box, score, label in zip(boxes, scores, labels):
                    left, top, right, bottom = box
                    if score > 0.0001:
                        output_file.write(
                            '{:s} {:.2f} {:d} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                                label, -1, -1, -10,  # type, truncated, occluded, alpha
                                left, top, right, bottom,
                                # bbox: left, top, right, bottom
                                -1, -1, -1,  # dimensions: height, width, length
                                -1, -1, -1,  # location: x,y,z
                                -10, np.log(score)# rotation_y, score
                            ))
                    # print('{}: {} {} {} {} Score: {}'.format(label, left, top, right, bottom, score))
            print('Saved results to ', file_name)

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions_multi,
                    output_folder=output_folder,
                    **extra_args)
