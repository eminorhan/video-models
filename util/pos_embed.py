# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import torch

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(checkpoint_model, orig_size, new_size):

    pos_tokens = checkpoint_model["pos_embed_spatial"]
    embedding_size = pos_tokens.shape[-1]
    print("Old shape:", pos_tokens.shape)

    # only the position tokens are interpolated
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed_spatial"] = pos_tokens
    print("New shape:", pos_tokens.shape)
