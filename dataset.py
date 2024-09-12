"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os.path
from PIL import Image
import glob

import torch.utils.data as data
import torch
import torchvision
import numpy as np
import random
from PIL import ImageFilter, ImageOps


def default_loader(path):
    return Image.open(path).convert('RGB')


class Dataset(data.Dataset):
    def __init__(self, root, image_size, transform=None, loader=default_loader, train=True, return_paths=False):
        super(Dataset, self).__init__()
        self.root = root
        if train:
            self.im_list = glob.glob(os.path.join(self.root, 'train', '*/*.jpg'))
            self.class_dir = glob.glob(os.path.join(self.root, 'train', '*'))
        else:
            self.im_list = glob.glob(os.path.join(self.root, 'test', '*/*.jpg'))
            self.class_dir = glob.glob(os.path.join(self.root, 'test', '*'))

        self.transform = transform
        self.loader = loader

        self.imgs = [(im_path, self.class_dir.index(os.path.dirname(im_path))) for
                     im_path in self.im_list]
        random.shuffle(self.imgs)

        self.return_paths = return_paths
        self.train = train
        self.image_size = image_size

        print('Seceed loading dataset!')

    def __getitem__(self, index):
        img_path, label = self.imgs[index]

        # image and its flipped image
        seg_path = img_path.split('_')[0]+'_mask.png'
        tex_path = img_path.split('_')[0] + '_00.jpg'
        img = self.loader(img_path)
        seg = Image.open(seg_path)
        tex = Image.open(tex_path)

        seg = seg.point(lambda p: p > 160 and 255)

        edge = seg.filter(ImageFilter.FIND_EDGES)
        edge = edge.filter(ImageFilter.SMOOTH_MORE)
        edge = edge.point(lambda p: p > 20 and 255)
        edge = torchvision.transforms.functional.to_tensor(edge).max(0, True)[0]

        img = torchvision.transforms.functional.to_tensor(img)
        seg = torchvision.transforms.functional.to_tensor(seg)
        tex = torchvision.transforms.functional.to_tensor(tex)
        seg_pro = seg[:3, :, :].max(0, True)[0]  # 改成对RGB三个通道取最大值
        seg_pro = (seg_pro > 0.5).float()
        img = img * seg_pro + torch.ones_like(img) * (1 - seg_pro)
        tex=tex * seg_pro + torch.ones_like(tex) * (1 - seg_pro)
        rgbs = torch.cat([img, seg_pro,tex], dim=0)
        img_non_tex = torch.cat([img, seg_pro], dim=0)

        data = {'images': rgbs, 'path': img_path, 'label': label,
                'edge': edge, 'image_non_tex': img_non_tex}

        return {'data': data}

    def __len__(self):
        return len(self.imgs)
