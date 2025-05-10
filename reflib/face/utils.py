import cv2
import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

class CFaceUtils:

    def __init__(self ):
        self.mydata = 0

    def img2tensor(self, imgs, bgr2rgb=True, float32=True):
        def _totensor(img, bgr2rgb, float32):
            if img.shape[2] == 3 and bgr2rgb:
                if img.dtype == 'float64':
                    img = img.astype('float32')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
            if float32:
                img = img.float()
            return img

        if isinstance(imgs, list):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        else:
            return _totensor(imgs, bgr2rgb, float32)

    def checkgray(self, img, threshold=10):
        img = Image.fromarray(img)
        if len(img.getbands()) == 1:
            return True
        img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
        img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
        img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
        diff1 = (img1 - img2).var()
        diff2 = (img2 - img3).var()
        diff3 = (img3 - img1).var()
        diff_sum = (diff1 + diff2 + diff3) / 3.0
        if diff_sum <= threshold:
            return True
        else:
            return False
