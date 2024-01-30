
import os
import math
import random
import argparse
import av
import torch
import numpy as np
from torchvision.utils import save_image
import models_vit, models_vit_img
from util.decoder.utils import tensor_normalize, spatial_sampling
from util.pos_embed import interpolate_pos_embed

MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

def get_args_parser():
    parser = argparse.ArgumentParser("Visualize ViT attention", add_help=False)
    parser.add_argument("--video_dir", default="demo", type=str, help="video directory where the video files are kept")
    parser.add_argument("--num_vids", default=1, type=int, help="Number of videos to do")
    parser.add_argument("--model_path", default="", type=str, help="path to pretrained model")
    parser.add_argument("--model_path_img", default="", type=str, help="path to pretrained image model")

    # Model parameters
    parser.add_argument("--model", default="vit_huge_patch14", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--drop_path_rate", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")

    # Augmentation parameters
    parser.add_argument("--color_jitter", type=float, default=None, metavar="PCT", help="Color jitter factor (enabled only when not using Auto/RandAug)")
    parser.add_argument("--aa", type=str, default="rand-m7-mstd0.5-inc1", metavar="NAME", help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument("--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split")

    # * Finetuning params
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool", help="Use class token instead of global pool for classification")
    parser.add_argument("--data_dirs", type=str, default=[""], nargs="+", help="Data paths")
    parser.add_argument("--datafile_dir", type=str, default="./datafiles", help="Store data files here")
    parser.add_argument("--output_dir", default="./embeddings", help="save embeddings here, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    # Video related configs
    parser.add_argument("--no_env", action="store_true")
    parser.add_argument("--rand_aug", default=False, action="store_true")
    parser.add_argument("--t_patch_size", default=2, type=int)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--checkpoint_period", default=1, type=int)
    parser.add_argument("--sampling_rate", default=2, type=int)
    parser.add_argument("--repeat_aug", default=1, type=int)
    parser.add_argument("--cpu_mix", action="store_true")
    parser.add_argument("--no_qkv_bias", action="store_true")
    parser.add_argument("--bias_wd", action="store_true")
    parser.add_argument("--sep_pos_embed", action="store_true")
    parser.set_defaults(sep_pos_embed=True)
    parser.add_argument("--fp32", action="store_true")
    parser.set_defaults(fp32=True)
    parser.add_argument("--jitter_scales_relative", default=[1.0, 1.0], type=float, nargs="+")
    parser.add_argument("--jitter_aspect_relative", default=[1.0, 1.0], type=float, nargs="+")
    parser.add_argument("--cls_embed", action="store_true")
    parser.set_defaults(cls_embed=True)

    return parser

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips_uniform, use_offset=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips_uniform == 1:
                # Take the center clip if num_clips_uniform is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips_uniform - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips_uniform
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx, start_idx / delta if delta != 0 else 0.0

def pyav_decode_stream(container, start_pts, end_pts, stream, stream_name, buffer_size=0):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts

def pyav_decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips_uniform=10,
    target_fps=30,
    use_offset=False,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Try to fetch the decoding information from the video head. Some videos do not support fetching the decoding information, in that case it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        clip_size = np.maximum(1.0, np.ceil(sampling_rate * (num_frames - 1) / target_fps * fps))
        start_idx, end_idx, fraction = get_start_end_idx(frames_length, clip_size, clip_idx, num_clips_uniform, use_offset=use_offset)
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(container, video_start_pts, video_end_pts, container.streams.video[0], {"video": 0})
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video

def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames

def prepare_video(path):
    video_container = av.open(path)
    frames, _, _ = pyav_decode(video_container, 4, 16, -1, num_clips_uniform=10, target_fps=30, use_offset=False)
    frames = temporal_sampling(frames, 0, 64, 16)
    frames = tensor_normalize(frames, torch.tensor(MEAN), torch.tensor(STD)).permute(3, 0, 1, 2)
    frames = spatial_sampling(
        frames,
        spatial_idx=1,
        min_scale=256,
        max_scale=256,
        crop_size=224,
        random_horizontal_flip=False,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
    )
    return frames

def list_subdirectories(directory):
    subdirectories = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            subdirectories.append(entry.path)
    subdirectories.sort()  # Sort the list of subdirectories alphabetically
    return subdirectories

def find_video_files(directory):
    """Recursively search for .mp4 or .webm files in a directory"""
    mp4_files = []
    subdir_idx = 0
    subdirectories = list_subdirectories(directory)
    for subdir in subdirectories:
        files = os.listdir(subdir)
        files.sort()
        for file in files:
            if file.endswith((".mp4", ".webm")):
                mp4_files.append(os.path.join(subdir, file))
        subdir_idx += 1
    return mp4_files

def interpolate_pos_embed_img(model, checkpoint_model):
    """Interpolate position embeddings for high-resolution."""
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    print(args)
    
    # set up and load video model
    model = models_vit.__dict__[args.model](img_size=args.input_size, **vars(args))
    checkpoint = torch.load(args.model_path, map_location='cpu')['model']
    # interpolate_pos_embed(checkpoint, 16, 32)  # last, interpolate position embedding
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    model.eval()
    # model.cuda()

    # set up and load image model
    model_img = models_vit_img.__dict__["vit_huge_patch14"](img_size=args.input_size, num_classes=0, global_pool=False)
    checkpoint_img = torch.load(args.model_path_img, map_location='cpu')['model']
    # interpolate_pos_embed_img(model_img, checkpoint_img)  # interpolate position embedding
    msg_img = model_img.load_state_dict(checkpoint_img, strict=False)
    print(msg_img)
    model_img.eval()
    # model_img.cuda()

    video_files = find_video_files(directory=args.video_dir)
    selected_files = random.sample(video_files, args.num_vids)
    print('Selected video files:', selected_files)

    for v in selected_files:
        vid = prepare_video(v)
        # vid = vid.cuda()
        img = vid.permute(1, 0, 2, 3)
        vid = vid.unsqueeze(0)

        with torch.no_grad():

            # video attention
            attn = model.get_last_selfattention(vid)
            attn = attn.squeeze(0)
            attn = attn[:, 0, 1:]
            attn = attn.view([16, 8, 16, 16]) # last two
            attn = torch.mean(attn, 0)
            attn = attn.unsqueeze(1)
            attn = attn.repeat(1, 3, 1, 1)
            attn = torch.nn.functional.interpolate(attn, size=(224, 224), mode='nearest-exact')
            print('Attn Vid shape:', attn.shape)

            # image attention
            attn_img = model_img.get_last_selfattention(img)
            attn_img = torch.mean(attn_img, 1)
            attn_img = attn_img[:, 0, 1:]
            attn_img = attn_img.view([16, 16, 16]) # last two
            attn_img = attn_img.unsqueeze(1)
            attn_img = attn_img.repeat(1, 3, 1, 1)
            attn_img = torch.nn.functional.interpolate(attn_img, size=(224, 224), mode='nearest-exact')
            attn_img = attn_img[::2, ...]
            print('Attn Img shape:', attn_img.shape)

            vid = vid.squeeze(0).permute(1, 0, 2, 3)
            vid = vid[::2, ...]
            vid = torch.nn.functional.interpolate(vid, size=(224, 224), mode='nearest-exact')
            print('Vid shape:', vid.shape)

            # stack vid and attn
            vid_attn = torch.cat((vid, attn, attn_img), 0)
            print('Vid-AttnVid-AttnImg shape:', vid_attn.shape)

            # save original image and attention map
            save_image(vid_attn, f'{os.path.splitext(os.path.basename(v))[0]}_vid_attn.jpg', nrow=8, padding=1, normalize=True, scale_each=True)