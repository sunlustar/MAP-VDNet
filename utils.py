import os
import torch
import numpy as np
from skimage.measure import compare_psnr


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)

def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])


def save_checkpoint(net, optimizer, epoch, losses, savepath):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    save_json = {
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losses': losses
    }
    torch.save(save_json, savepath)


def load_checkpoint(net, optimizer, checkpoint_path):
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    losses = checkpoint['losses']

    return net, optimizer, start_epoch, losses


