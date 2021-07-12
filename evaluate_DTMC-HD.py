import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from skimage.measure import compare_ssim, compare_psnr
from utils import mkdir_if_not_exist

############ Parameters ###############
N = 7
sigma = 50
scale = 16
cuda_flag = True

SAVE_PNG = False
savedir = './results/'
dataset_dir = './testsets/DTMC-HD/'
psnrsave = 'ASU_1080p_%d.txt' % sigma
model_path = './mynetwork/denoising_%d.pkl' % sigma


def evaluate(dataset_dir, out_img_dir, cuda_flag=True, savetxt='psnr.txt'):

    denoisenet = torch.load(model_path)
    if isinstance(denoisenet, torch.nn.DataParallel):
        denoisenet = denoisenet.module

    if cuda_flag:
        denoisenet.cuda().eval()
    else:
        denoisenet.eval()

    if SAVE_PNG:
        mkdir_if_not_exist(out_img_dir)
        mkdir_if_not_exist(out_img_dir + 'DTMC-HD/')
        mkdir_if_not_exist(out_img_dir + 'DTMC-HD/%d/' % sigma)

    test_img_list = os.listdir(dataset_dir)
    str_format = 'im%d.png'

    total_count = 483 + 7*6
    count = 0


    pre = datetime.datetime.now()
    psnr_val = 0
    ssim_val = 0

    for file in test_img_list:
        seq_noisy = dataset_dir + file + '/Gauss_%d/noisy/' %sigma
        num = len(os.listdir(seq_noisy))
        psnr = 0
        ssim = 0

        if SAVE_PNG:
            mkdir_if_not_exist(out_img_dir + 'DTMC-HD/%d/%s/' % (sigma, file))

        noisy_frames = []
        for j in range(1, num+1):
            noisy_path = seq_noisy + 'im%d.png' % j
            noisy_frames.append(plt.imread(noisy_path))

        ######### pad ########
        h, w, c = noisy_frames[0].shape
        nh = (h//scale + 1) * scale
        noisy_frames = np.array(noisy_frames)
        noisy_frames_padded = np.lib.pad(noisy_frames, pad_width=((N // 2, N // 2), ((nh-h)//2, (nh-h)//2),
                                                                  (0, 0), (0, 0)), mode='constant')
        noisy_frames_padded = np.transpose(noisy_frames_padded, (0, 3, 1, 2))

        for i in range(num):
            reference_path = dataset_dir + file + '/ori/' + str_format % (i+1)
            reference_frame = plt.imread(reference_path)

            input_frames = noisy_frames_padded[i:i+N]
            input_frames = torch.from_numpy(input_frames).cuda()

            input_frames = input_frames.view(1, input_frames.size(0), input_frames.size(1), input_frames.size(2), input_frames.size(3))

            x_list = denoisenet(input_frames)
            predicted_img = x_list[-1][0, :, (nh-h)//2:-(nh-h)//2]

            Img = predicted_img.permute(1, 2, 0).data.cpu().numpy().astype(np.float32)

            count += 1

            ######## compare PSNR and SSIM ########
            psnr += compare_psnr(Img, reference_frame, data_range=1.)
            ssim += compare_ssim(Img, reference_frame, data_range=1., multichannel=True)

            ######## save output images ########
            if SAVE_PNG:
                plt.imsave(out_img_dir + 'DTMC-HD/%d/%s/im%d.png' % (sigma, file, i+1), np.clip(Img, 0.0, 1.0))

            cur = datetime.datetime.now()
            processing_time = (cur - pre).seconds / count
            print('%.2fs per frame.\t%.2fs left.' % (processing_time, processing_time * (total_count - count)))

        print('video %s, psnr %.4f, ssim %.4f.\n' % (file, psnr/num, ssim/num))
        psnr_val += psnr / num
        ssim_val += ssim / num
        # save loss.txt
        txtfile = open(savetxt, 'a')
        txtfile.write('video %s, psnr %.4f, ssim %.4f.\n' % (file, psnr/num, ssim/num))
        txtfile.close()

    ave_psnr_val = psnr_val / len(test_img_list)
    ave_ssim_val = ssim_val / len(test_img_list)
    print('PSNR_val: %.4fdB, SSIM_val: %.4f.\n' % (ave_psnr_val, ave_ssim_val))
    txtfile = open(savetxt, 'a')
    txtfile.write('Average psnr %.4f, ssim %.4f.\n\n' % (ave_psnr_val, ave_ssim_val))
    txtfile.close()

with torch.no_grad():
    evaluate(dataset_dir, savedir, cuda_flag=cuda_flag, savetxt=psnrsave)
False