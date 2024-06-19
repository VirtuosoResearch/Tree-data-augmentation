import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

class LISAMixUp:
    """
    Unlike MixUp within-y, which mixes data with other points in a batch, LISA samples NEW data from the dataset to mix in 
    with the batch, meaning it may use twice the number of data points as normal MixUp.
    """
    def __init__(self, mix_list, alpha=0.5, split="train"):
        assert alpha > 0
        self.mix_list = mix_list
        self.alpha = alpha
        self.split = split
        self.nargs = 2

    def __call__(self, x):
        x = to_tensor(x)
        lmbda = np.random.beta(self.alpha, self.alpha)
        x2 = np.random.choice(self.mix_list)
        return lmbda * x + (1-lmbda) * x2
                