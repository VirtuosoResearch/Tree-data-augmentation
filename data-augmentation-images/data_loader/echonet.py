from torch.utils.data import Dataset
import os
import torch
from PIL import Image

class EchonetDataset(Dataset):
    def __init__(self, data_dir, transform=None, return_index=False):
        self.data_dir = os.path.join(data_dir, "echonet/data/images")
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return int(len(os.listdir(self.data_dir))/2)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, "%d.png" % index)
        label = torch.load(os.path.join(self.data_dir, "seg_labels_%d.pt" % index))
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img) if self.transform is not None else img
        img = (img[0].permute(0, 2, 1), img[1].permute(0, 2, 1)) if len(img) == 2 else img.permute(2, 1, 0)
        
        if self.return_index:
            return img, label, index
        
        return img, label