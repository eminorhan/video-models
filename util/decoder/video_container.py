# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


def get_video_container(path_to_vid):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
    Returns:
        container (container): video container.
    """
    with open(path_to_vid, "rb") as fp:
        container = fp.read()
    return container