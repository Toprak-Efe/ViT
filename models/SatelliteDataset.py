import glob
import os
from torch.utils.data import Dataset 
from PIL import Image

class SatelliteDataset(Dataset):
  def __init__(self, image_folder, truth_folder, transform_data=None, transform_truth=None):
    path = r'C:\Users\topra\Documents\Jupyter\Satellite-Model\ViT\data'
    image_path = os.path.join(path, image_folder, '*.png')
    truth_path = os.path.join(path, truth_folder, '*.png')
    image_names_cand = glob.glob(image_path)
    truth_images_cand = glob.glob(truth_path)
    self.image_names = []
    self.truth_images = []
    for image_name, i in zip(image_names_cand, range(len(image_names_cand))):
      truth_name = image_name.replace('images', 'truth')
      if truth_name in truth_images_cand:
        self.image_names.append(image_name)
        self.truth_images.append(truth_name)
    self.transform_data = transform_data
    self.transform_truth = transform_truth

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    image = Image.open(self.image_names[idx])
    truth = Image.open(self.truth_images[idx])
    if self.transform_data:
      image = self.transform_data(image)
    if self.transform_truth:
      truth = self.transform_truth(truth)
    return image, truth