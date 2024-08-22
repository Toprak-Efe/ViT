import torch
import torch.nn as nn
from models.PatchPositionalEmbedding import PatchPositionalEmbedding
from models.PatchPositionalExtractor import PatchPositionalExtractor

class Model(nn.Module):
  def __init__(self,
               patch_size: int=8,
               channels: int=3):
    super(Model, self).__init__()
    self.embed = PatchPositionalEmbedding(channels=channels, patch_size=patch_size)
    self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=patch_size*patch_size*channels, nhead=8), num_layers=6)
    self.extract = PatchPositionalExtractor(channels=channels, patch_size=patch_size, output_size=512)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.embed(x)
    x = self.transformer(x)
    x = self.extract(x)
    return x.sum(dim=1)
