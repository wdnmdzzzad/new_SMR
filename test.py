import os.path
import sys

from PIL import Image
import glob

import torch.utils.data as data
import torch
import torchvision
import numpy as np
import random
from PIL import ImageFilter, ImageOps
torch.set_printoptions(threshold=np.inf)


root_path='E:/All_application_resouces/test/output/output1_gray.jpg'
seg_path = root_path.replace('.jpg', '.png')
img = Image.open(root_path)
seg = Image.open(seg_path)


img = torchvision.transforms.functional.to_tensor(img)
seg_1=torchvision.transforms.functional.to_tensor(seg)
#print(seg_1)

seg = torchvision.transforms.functional.to_tensor(seg).max(0, True)[0]

#print(img)
#print(seg)

img = img * seg + torch.ones_like(img) * (1 - seg)
#print(img)
rgbs = torch.cat([img, seg], dim=0)
print(rgbs)
