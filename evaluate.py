import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from utils import mkdir_if_not_exist
from skimage.measure import compare_ssim, compare_psnr
import warnings
warnings.filterwarnings("ignore", module="matplotlib.pyplot")
# ------------------------------
plt.switch_backend('agg')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sigma = 15
pathlistfile = './list/test_list.txt'
model_path = './mynetwork/denoising_%d.pkl' % sigma
cuda_flag = True
SAVE_PNG = True
savedir = './results/Vimeo/'

### Please replace with your path of Vimeo-90K dataset ###
ground_truth_dir = 'I:/Datasets/vimeo_septuplet/sequences/'
dataset_dir = 'I:/Datasets/vimeo_septuplet/sequences_with_noise_%d/' % sigma

def vimeo_evaluate(input_dir, gt_dir, out_img_dir, test_codelistfile, cuda_flag=True):

    denoisenet = torch.load(model_path)
    if isinstance(denoisenet, torch.nn.DataParallel):
        denoisenet = denoisenet.module

    if cuda_flag:
        denoisenet.cuda().eval()
    else:
        denoisenet.eval()

    fp = open(test_codelistfile)
    test_img_list = fp.read().splitlines()
    fp.close()

    if SAVE_PNG:
        mkdir_if_not_exist(savedir)

    process_index = [1, 2, 3, 4, 5, 6, 7]
    str_format = 'im%04d.png'

    total_count = len(test_img_list)
    count = 0

    pre = datetime.datetime.now()
    psnr_val = 0
    ssim_val = 0
    for code in test_img_list:
        count += 1

        input_frames = []
        for i in process_index:
            input_frames.append(plt.imread(os.path.join(input_dir, code, str_format % i)))
        input_frames = np.transpose(np.array(input_frames), (0, 3, 1, 2))

        gt_path = gt_dir + '/' + code + '/' + 'im4.png'
        reference_frame = plt.imread(gt_path)
        reference_frame = torch.Tensor(reference_frame.transpose(2, 0, 1))

        if cuda_flag:
            input_frames = torch.from_numpy(input_frames).cuda()
        else:
            input_frames = torch.from_numpy(input_frames)
        input_frames = input_frames.view(1, input_frames.size(0), input_frames.size(1), input_frames.size(2), input_frames.size(3))

        out = denoisenet(input_frames)
        predicted_img = out[-1][0, :, :, :]

        Img = predicted_img.permute(1, 2, 0).data.cpu().numpy().astype(np.float32)
        Iclean = reference_frame.permute(1, 2, 0).data.cpu().numpy().astype(np.float32)

        ######## compare PSNR and SSIM ########
        psnr_val += compare_psnr(Img, Iclean, data_range=1.)
        ssim_val += compare_ssim(Img, Iclean, data_range=1., multichannel=True)

        ######## save output images ########
        if SAVE_PNG:
            video = code.split('/')[0]
            sep = code.split('/')[1]
            mkdir_if_not_exist(os.path.join(out_img_dir, video))
            mkdir_if_not_exist(os.path.join(out_img_dir, video, sep))
            plt.imsave(os.path.join(out_img_dir, video, sep, 'im4.png'), np.clip(Img, 0.0, 1.0))

        cur = datetime.datetime.now()
        processing_time = (cur - pre).seconds / count
        print('%.2fs per frame.\t%.2fs left.' % (processing_time, processing_time * (total_count - count)))

    ave_psnr_val = psnr_val / count
    ave_ssim_val = ssim_val / count
    print('PSNR_val: %.4fdB, SSIM_val: %.4fdB\n' % (ave_psnr_val, ave_ssim_val))

with torch.no_grad():
    vimeo_evaluate(dataset_dir, ground_truth_dir, savedir, pathlistfile, cuda_flag=cuda_flag)
False