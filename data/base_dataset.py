### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @classmethod
    def name(cls):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    cen = opt.fineSize/2
    x = (new_w/2)-cen
    y = (new_h/2)-cen
    #x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    #y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'auto' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __auto_resize(img, opt.threshold / opt.batchSize, method)))
        base = float(2 ** opt.n_downsample_global)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))
    elif 'resize_w_h' in opt.resize_or_crop:
        osize = [opt.width, opt.height]
        transform_list.append(transforms.Resize(osize, method))
    elif 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width_p2' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))
        base = float(2 ** opt.n_downsample_global)
        transform_list.append(transforms.Lambda(
            lambda img: __make_power_2(img, base, method)))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(
            lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __auto_resize(img, threshold, method=Image.BICUBIC):
    ow, oh = img.size
    if ow * oh > threshold:
        factor = (threshold / (ow * oh)) ** 0.5
        w = int(ow * factor)
        h = int(oh * factor)
        return img.resize((w, h), method)
    return img


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
