import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchPositionalEmbedding(nn.Module):
  def __init__(self,
               channels: int=3,
               patch_size: int=128,
               pos_embed_size: int=512):
    super(PatchPositionalEmbedding, self).__init__()
    self.channels = channels
    self.patch_size = patch_size
    self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0) # (batch, C*patch_size*patch_size, num_tokens)
    self.embed = nn.Parameter(torch.randn(1, pos_embed_size, channels*patch_size*patch_size)) # (batch, token_len, C*patch_size*patch_size)
      
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.unfold(x)
    x = torch.concat([x.permute(0, 2, 1), torch.zeros(x.size(0), 1, self.channels*self.patch_size*self.patch_size).to(x.device)], dim=1)
    embed = F.interpolate(self.embed.permute(0, 2, 1), size=x.size(1), mode='linear', align_corners=False).permute(0, 2, 1) # (batch, token_len+1, C*patch_size*patch_size)
    return x + embed