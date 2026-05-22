import torch
from einops import rearrange, einsum
import einx
## Basic implementation
# Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?
## Einsum is self-documenting and robust
#
# Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
## Or, a batched version where D can have any leading dimensions but A is constrained.
# Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")

images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value
## Or in one go:
dimmed_images = einsum(
images, dim_by,
"batch height width channel, dim_value -> batch dim_value height width channel"
)

channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)
# Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3))
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
## Instead, using einops:
height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
channels_last,
"batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
channels_first, B,
"batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
channels_first_transformed,
"batch channel (height width) -> batch height width channel",
height=height, width=width
)
## Or, if you’re feeling crazy: all in one go using einx.dot (einx equivalent of einops.einsum)
height = width = 32
channels_last_transformed = einx.dot(
"batch row_in col_in channel, (row_out col_out) (row_in col_in)"
"-> batch row_out col_out channel",
channels_last, B,
col_in=width, col_out=width
)