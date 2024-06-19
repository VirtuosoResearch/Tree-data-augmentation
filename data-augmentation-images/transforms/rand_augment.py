import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
import random
import torchvision

def _ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

class ShearX:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _ShearX(img, val)

def _ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

class ShearY:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _ShearY(img, val)

def _TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

class TranslateX:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _TranslateX(img, val) 


def _TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

class TranslateXabs:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _TranslateXabs(img, val)

def _TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

class TranslateY:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _TranslateY(img, val)

def _TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

class TranslateYabs:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _TranslateYabs(img, val)

def _Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)

class Rotate:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Rotate(img, val)

def _AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

class AutoContrast:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _AutoContrast(img, val)

def _Invert(img, _):
    return PIL.ImageOps.invert(img)

class Invert:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Invert(img, val)


def _Equalize(img, _):
    return PIL.ImageOps.equalize(img)

class Equalize:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Equalize(img, val)

def _Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)

class Flip:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Flip(img, val)

def _Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)

class Solarize:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Solarize(img, val)

def _SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)

class SolarizeAdd:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _SolarizeAdd(img, addition=val, threshold=128)

def _Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

class Posterize:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Posterize(img, val)


def _Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)

class Contrast:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Contrast(img, val)

def _Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)

class Color:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Color(img, val)

def _Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)

class Brightness:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Brightness(img, val)

def _Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

class Sharpness:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Sharpness(img, val)

def _Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return _CutoutAbs(img, v)

class Cutout:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _Cutout(img, val)

def _CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

class CutoutAbs:

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        tmp_m = np.random.randint(0, 31)
        val = (float(tmp_m) / 30) * float(self.maxval - self.minval) + self.minval
        return _CutoutAbs(img, val)

def _Identity(img, v=0):
    return img

class Identity: 

    def __init__(self, maxval, minval):
        self.maxval = maxval
        self.minval = minval

    def __call__(self, img):
        return _Identity(img)

augmentations = {
    "Identity": (Identity, 0, 1),
    "AutoContrast": (AutoContrast, 0, 1),
    "Equalize": (Equalize, 0, 1),
    "Invert": (Invert, 0, 1), # 
    "Rotate": (Rotate, 0, 30),
    "Posterize": (Posterize, 0, 4),
    "Solarize": (Solarize, 0, 256),
    "SolarizeAdd": (SolarizeAdd, 0, 110), # 
    "Color": (Color, 0.1, 1.9),
    "Contrast": (Contrast, 0.1, 1.9),
    "Brightness": (Brightness, 0.1, 1.9),
    "Sharpness": (Sharpness, 0.1, 1.9),
    "ShearX": (ShearX, 0., 0.3),
    "ShearY": (ShearY, 0., 0.3),
    "Cutout": (CutoutAbs, 0, 40), # 
    "TranslateXabs": (TranslateXabs, 0., 100),
    "TranslateYabs": (TranslateYabs, 0., 100),
}

augmentation_list = [
        (AutoContrast, 0, 1),
        (Equalize, 0, 1),
        (Invert, 0, 1),
        (Rotate, 0, 30),
        (Posterize, 0, 4),
        (Solarize, 0, 256),
        (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        (ShearX, 0., 0.3),
        (ShearY, 0., 0.3),
        (CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
    ]

class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augmentation_list

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(val, val)(img)

        return img


class RandomAugmentation:
    '''Deprecated'''
    def __init__(self, n, seed = 0, size=224, **kwargs):
        self.n = n # number of augmentations

        color_jitter = torchvision.transforms.ColorJitter(
            0.8, 0.8, 0.8, 0.2
        )
        self.augmentations = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
            torchvision.transforms.RandomApply([color_jitter], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2)
        ]
        
        rng = np.random.default_rng(seed)
        ops = rng.choice(len(augmentation_list), size=self.n, replace=True)
        ops = [augmentation_list[op] for op in ops]
        for op, minval, maxval in ops:
            self.augmentations.append(op(maxval, minval))

        self.augmentations.append(torchvision.transforms.ToTensor())
        self.augmentations = torchvision.transforms.Compose(self.augmentations)

    def __call__(self, img):
        return self.augmentations(img), self.augmentations(img)

