# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import os
import torch

import helpers.misc as misc
import torch.backends.cudnn as cudnn

from pathlib import Path
from torch.utils.data import SequentialSampler, DataLoader
from utils import load_model
from helpers.kinetics import Kinetics

def get_args_parser():
    parser = argparse.ArgumentParser("Evaluate a pretrained model on video recognition", add_help=False)

    # Model parameters
    parser.add_argument("--model_name", default='vit_s_none', type=str, help='Model identifier')
    parser.add_argument("--img_size", default=224, type=int, help="images input size")

    # Bookkeeping
    parser.add_argument("--out_dir", default="outputs", help="dump junk here")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--num_workers", default=16, type=int)

    # Data parameters
    parser.add_argument("--val_dir", default="", help="path to val data")
    parser.add_argument("--val_mode", type=str, default="val", help="Which mode?")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--num_frames", default=16, type=int, help="Number of frames")
    parser.add_argument("--sampling_rate", default=8, type=int, help="Number of frames between consecutive samples (sampling period)")
    parser.add_argument("--jitter_scales_relative", default=[0.08, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[0.75, 1.3333], type=float, nargs="+")
    parser.add_argument("--train_jitter_scales", default=[256, 320], type=int, nargs="+")

    return parser

def list_subdirectories(directory):
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdirectories.append(entry.path)
    subdirectories.sort()  # Sort the list of subdirectories alphabetically
    return subdirectories

def find_mp4_files(directory):
    """Recursively search for .mp4 or .webm files in a directory"""
    mp4_files = []
    subdir_idx = 0
    subdirectories = list_subdirectories(directory)
    for subdir in subdirectories:
        files = os.listdir(subdir)
        files.sort()
        for file in files:
            if file.endswith((".mp4", ".webm")):
                mp4_files.append((os.path.join(subdir, file), subdir_idx))
        subdir_idx += 1
    return mp4_files

def write_csv(video_files, save_dir, save_name):
    """Write the .csv file with video path and subfolder index"""
    with open(os.path.join(save_dir, f'{save_name}.csv'), 'w', newline='') as csvfile:
        for video_file, subdir_idx in video_files:
            csvfile.write(f"{video_file}, {subdir_idx}\n")

def main(args):
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    cudnn.benchmark = True

    val_dataset = Kinetics(
        mode=args.val_mode,
        datafile_dir=args.out_dir,
        sampling_rate=args.sampling_rate,
        num_frames=args.num_frames,
        train_jitter_scales=tuple(args.train_jitter_scales),
        train_crop_size=args.img_size,
        test_crop_size=args.img_size,
        jitter_aspect_relative=args.jitter_aspect_relative,
        jitter_scales_relative=args.jitter_scales_relative,
    )

    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # set up and load model
    model = load_model(args.model_name)
    model.to(device)  # move model to device
    print(f"Model = {model}")

    # evaluate model and print results
    test_stats = evaluate(val_loader, model, device)
    print("==========================================")
    print(f"Number of test videos: {len(val_dataset)}")
    print(f"Acc@1: {test_stats['acc1']:.1f}%") 
    print(f"Acc@5: {test_stats['acc5']:.1f}%")
    print(f"Loss: {test_stats['loss']:.2f}")


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    num_logs_per_epoch = 1

    # switch to evaluation mode
    model.eval()

    for _, (images, target) in enumerate(metric_logger.log_every(data_loader, len(data_loader) // num_logs_per_epoch, header)):

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if len(images.shape) == 6:
            b, r, c, t, h, w = images.shape
            images = images.view(b * r, c, t, h, w)
            target = target.view(b * r)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = misc.accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # prepare data file
    val_files = find_mp4_files(directory=args.val_dir)
    write_csv(video_files=val_files, save_dir=args.out_dir, save_name=args.val_mode)

    # evaluate
    main(args)