import torch
import torch.nn as nn

class PatchPositionalExtractor(nn.Module):
  def __init__(self,
               channels: int=3,
               patch_size: int=128,
               output_size: int=512):
    super(PatchPositionalExtractor, self).__init__()
    self.channels = channels
    self.patch_size = patch_size
    self.output_size = output_size
    self.fold = nn.Fold(output_size, kernel_size=patch_size, stride=patch_size) # (batch, C*patch_size*patch_size, num_tokens)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    token_len = x.size(1) - 1
    assert self.output_size % self.patch_size == 0, "Output size must be divisible by patch size"
    x = x[:, :(self.output_size**2)//(self.patch_size**2), :] # (batch, num_tokens, C*patch_size*patch_size)
    x = self.fold(x.permute(0, 2, 1)) # (batch, C*patch_size*patch_size, output_size**2)
    return x