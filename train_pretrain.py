import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Network_Alignment import Alignment
import matplotlib.pyplot as plt
from read_data_clip import MemoryFriendlyLoader
from utils import batch_PSNR, load_checkpoint, save_checkpoint, show_time

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

def main(args):
    # Load dataset
    print('> Loading dataset ...')
    Dataset = MemoryFriendlyLoader(origin_img_dir=args.gt_dir, edited_img_dir=args.train_dir,
                                   pathlistfile=args.filelist)
    loader_train = torch.utils.data.DataLoader(dataset=Dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True)
    print('\t# of training samples: %d\n' % int(len(Dataset)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = Alignment(cuda_flag=True).cuda()
    model = torch.nn.DataParallel(model)

    criterion = nn.L1Loss().cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    plotx = []
    ploty = []

    checkpoint_path = args.ckp_dir + 'checkpoint_%depoch_%d_pretrain.ckpt'\
                      % (args.start_epoch, args.sigma)
    if args.use_checkpoint:
        model, optimizer, start_epoch, ploty = load_checkpoint(model, optimizer, checkpoint_path)
        model = torch.nn.DataParallel(model)
        print('cps loaded!')
        plotx = list(range(len(ploty)))


    # Training
    for epoch in range(args.start_epoch, args.epochs):
        losses = 0
        # train over all data in the epoch
        for step, (x, y, path_code) in enumerate(loader_train):

            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            frames_input = x
            reference = y

            frames_input = Variable(frames_input.cuda())
            reference = Variable(reference.cuda())

            # Evaluate model and optimize it
            out_train = model(frames_input)
            loss = 0
            index = [0, 1, 2, 4, 5, 6]
            for i in index:
                loss += criterion(out_train[:, i, :, :, :], reference)
            losses += loss.item()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                # Results
                model.eval()
                psnr_train = 0
                for i in index:
                    psnr_train += batch_PSNR(out_train[:, i, :, :, :], reference, 1.)
                psnr_train = psnr_train / 6
                print('%s  [epoch %d][%d/%d]  loss: %f  PSNR_train: %.4fdB' % \
                    (show_time(datetime.datetime.now()), epoch + 1, step + 1, len(loader_train), losses / (step+1), psnr_train))

        # save loss pic
        plotx.append(epoch + 1)
        ploty.append(losses / (step + 1))
        if epoch // 1 == epoch / 1:
            plt.plot(plotx, ploty)
            plt.savefig(args.loss_pic)
        # save loss.txt
        file = open(args.savetxt, 'a')
        file.write('epoch %d loss: %f, val_psnr: %f\n' % ((epoch + 1), losses / (step+1), psnr_train))
        file.close()
        # save checkpoint
        if not os.path.exists(args.ckp_dir):
            os.mkdir(args.ckp_dir)
        save_checkpoint(model, optimizer, epoch + 1, ploty, args.ckp_dir + 'checkpoint_%depoch_%d_pretrain.ckpt'
                        % (epoch + 1, args.sigma))
        # save align.pkl
        torch.save(model, os.path.join(args.save_dir + '/align_%d.pkl' % args.sigma))
        torch.save(model.module.state_dict(), os.path.join(args.save_dir + '/align_%d.pth' % args.sigma))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VideoDenoising Training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--loss_pic', type=str, default='loss_pic_pretrain.png')
    parser.add_argument('--savetxt', type=str, default='loss_pretrain.txt')
    parser.add_argument('--save_dir', type=str, default='./mynetwork/')
    parser.add_argument('--ckp_dir', type=str, default='./checkpoints/')
    parser.add_argument('--filelist', type=str, default='./list/train_list.txt')
    parser.add_argument('--color', default=True, help='Train with color instead of grayscale')
    parser.add_argument('--use_checkpoint', default=False)
    parser.add_argument('--start_epoch', type=int, default=0, help='Number of start training epoch')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')

    ### change with your settings ###
    parser.add_argument('--sigma', type=int, default=25, help='Simulated noise level')
    parser.add_argument('--train_dir', type=str, default='I:/Datasets/vimeo_septuplet/sequences_with_noise_')
    parser.add_argument('--gt_dir', type=str, default='I:/Datasets/vimeo_septuplet/sequences_with_noise_')

    args = parser.parse_args()

    # modify some parameters
    args.loss_pic = args.loss_pic[:-4] + '_%d.png' % args.sigma
    args.savetxt = args.savetxt[:-4] + '_%d.txt' % args.sigma
    args.train_dir = args.train_dir + '%d/' % args.sigma
    args.gt_dir = args.gt_dir + '%d/' % args.sigma

    print(args)

    main(args)
