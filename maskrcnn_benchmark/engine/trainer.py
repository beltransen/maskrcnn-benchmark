# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import random
import cv2
import numpy as np
import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from tensorboardX import SummaryWriter

from apex import amp

random.seed(0)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")

    # Create Tensorboard logger
    board_writer = SummaryWriter(arguments['log_dir'])

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        # Visualize images in input tensor
        # for f in range(images.tensors.shape[2]):
        #     img = images.tensors[0,:,f,:,:].numpy()
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     img = np.moveaxis(img, 0, -1)
        #     cv2.imshow('frame ', img)
        #     cv2.waitKey(0)


        # Visualize ground truth annotations
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        #
        # if len(images.tensors.shape) == 5:
        #     num_images = images.tensors.shape[2]
        # else:
        #     num_images = images.tensors.shape[0]
        #
        # for f in range(num_images):
        #     if len(images.tensors.shape) == 5:
        #         img = images.tensors[0,:,f,:,:].numpy()
        #     else:
        #         img = images.tensors[f,:,:,:].numpy()
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     img = np.moveaxis(img, 0, -1)
        #     labels = targets[0].get_field("labels")
        #     boxes = targets[0].bbox
        #
        #     palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        #     colors = labels[:, None] * palette
        #     colors = (colors % 255).numpy().astype("uint8")
        #     colors = colors.tolist()
        #
        #     for box, color in zip(boxes, colors):
        #         box = box.to(torch.int64)
        #         top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        #         img = cv2.rectangle(
        #             img, tuple(top_left), tuple(bottom_right), tuple(color), 5
        #
        #         )
        #
        #     cv2.imshow('labels', img)
        #     cv2.waitKey(0)

        images = images.to(device)

        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            # Write iter metrics to TensorboardX logger
            info = {}
            for name, meter in meters.meters.items():
                info[name] = meter.global_avg
            board_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], iteration)
            board_writer.add_scalars('loss', info, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
