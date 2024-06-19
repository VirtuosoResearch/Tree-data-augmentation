from typing import Any
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from PIL import ImageFilter
import numpy as np
import torch
from torch import nn
import random
import torchvision
from torchvision.transforms import GaussianBlur, Compose
from transforms.rand_augment import CutoutAbs, Rotate

color_jitter = torchvision.transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.2
        )

class Sobel():

    def __call__(self, image) -> Any:
        image = image.convert("L")
 
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        image = image.filter(ImageFilter.FIND_EDGES)
        return image.convert("RGB")
        
class GaussianNoise():
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self, img):
        out = img + self.sigma * np.random.randn(*np.copy(np.array(img)).shape).astype(np.float32)
        return out


sim_augmentations = {
    "Crop":   torchvision.transforms.RandomResizedCrop(size=224),
    "Cutout": CutoutAbs(20, 10),
    "Color":  color_jitter, # torchvision.transforms.RandomApply([color_jitter], p=0.8),
    "Sobel":  Sobel(), # torchvision.transforms.RandomApply([Sobel()], p=0.8),
    "Noise":  GaussianNoise(0.0001),
    "Blur":   GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.)),
    "Rotate": Rotate(20, 10)
}

class SimCLRTestTransfrom:

    def __init__(self, transform_names=[], ratios=[], pre_transforms=[], post_transforms=[], 
                 probs=None, tree_idxes=None, **kwargs):
        '''
        A generic class for simclr data augmentation that takes in a list of transform names and ratios
        '''
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.transforms = []
        ''' Implement all other augmentations '''
        ops = [sim_augmentations[name] for name in transform_names]
        for i, op in enumerate(ops):
            # assert ratios[i] in [0.2, 0.4, 0.6, 0.8, 1.0]
            # tmp_min, tmp_max = minval + (ratios[i]-0.2)*(maxval - minval), minval + (ratios[i])*(maxval - minval)
            self.transforms.append(op)

        self.probs = probs

        self.max_idx = np.max(tree_idxes)+1 if tree_idxes is not None else None
        self.tree_idxes = tree_idxes
        self.idx_to_transform = self.generate_idx_to_transforms(tree_idxes)
    
    def generate_idx_to_transforms(self, tree_idxes):
        if tree_idxes is None:
            return None
        transformed_idxes = np.ones(self.max_idx)*-1
        for i, idx in enumerate(tree_idxes):
            transformed_idxes[idx] = i
        return transformed_idxes.astype(int)
    
    def add_pre_and_post(self, selected_transforms):
        final_transforms = self.pre_transforms[:]
        final_transforms += selected_transforms
        final_transforms.append(torchvision.transforms.ToTensor())
        final_transforms += self.post_transforms
        # print(final_transforms)
        return Compose(final_transforms)

    def apply_transform(self, data):
        if self.probs is None:
            return self.add_pre_and_post(self.transforms)(data)
        elif self.tree_idxes is None:
            tem_transform = []
            for prob, transform in zip(self.probs, self.transforms):
                if random.random() < prob:
                    tem_transform.append(transform)
                    # data = self.transform(data)
            return self.add_pre_and_post(tem_transform)(data)
        else: 
            cur_idx = 0
            tem_transform = []
            cur_prob = self.probs[self.idx_to_transform[cur_idx]]
            if random.random() >= cur_prob: # probability of index 0
                return data

            # appy transforms based on the tree
            while cur_idx < self.max_idx:
                if self.idx_to_transform[cur_idx] == -1:
                    break
                cur_transform = self.transforms[self.idx_to_transform[cur_idx]]
                cur_prob = self.probs[self.idx_to_transform[cur_idx]]
                tem_transform.append(cur_transform)

                if (cur_idx*2 + 1) < len(self.idx_to_transform) and self.idx_to_transform[cur_idx*2+1] != -1:
                    next_idx = cur_idx*2+1
                elif (cur_idx*2 + 2) < len(self.idx_to_transform) and self.idx_to_transform[cur_idx*2+2] != -1:
                    next_idx = cur_idx*2+2
                else:
                    break

                next_prob = self.probs[self.idx_to_transform[next_idx]]
                if random.random() < next_prob:
                    cur_idx = cur_idx*2 + 1
                else:
                    cur_idx = cur_idx*2 + 2
                    
            return self.add_pre_and_post(tem_transform)(data)
    
    def __call__(self, data):
        return self.apply_transform(data), self.apply_transform(data)