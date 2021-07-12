import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import random


class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, origin_img_dir, pathlistfile, edited_img_dir=''):
        self.origin_img_dir = origin_img_dir
        self.edited_img_dir = edited_img_dir
        self.pathlist = self.loadpath(pathlistfile)
        self.count = len(self.pathlist)

    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        random.shuffle(pathlist)
        return pathlist

    def __getitem__(self, index):
        frames = []

        path_code = self.pathlist[index]
        N = 7
        ref = plt.imread(os.path.join(self.origin_img_dir, path_code, 'im0004.png'))
        h, w, c = ref.shape
        hh = 128
        ww = 224
        # print(h, w)
        raw = random.randint(0, h - hh - 1)
        col = random.randint(0, w - ww - 1)
        frames.append(ref[raw:raw+hh, col:col+ww, :])                   # load ground truth
        for i in range(7):
            img = plt.imread(os.path.join(self.edited_img_dir, path_code, 'im%04d.png' % (i + 1)))
            frames.append(img[raw:raw+hh, col:col+ww, :])

        frames = np.array(frames)
        framex = np.transpose(frames[1:8, :, :, :], (0, 3, 1, 2))
        framey = np.transpose(frames[0, :, :, :], (2, 0, 1))

        return torch.from_numpy(framex), torch.from_numpy(framey), path_code

    def __len__(self):
        return self.count

