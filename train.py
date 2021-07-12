import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from Network_MAPVDNet import MAPVDNet
import matplotlib.pyplot as plt
from read_data import MemoryFriendlyLoader
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

    model = MAPVDNet(cuda_flag=True, alignment_model=args.pretrained_model, T=args.stages).cuda()
    model = torch.nn.DataParallel(model)

    criterion = nn.L1Loss().cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    plotx = []
    ploty = []

    checkpoint_path = args.ckp_dir + 'checkpoint_%d_%depoch.ckpt' % (args.sigma, args.start_epoch)
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
            frame_clean = y

            frame_clean = Variable(frame_clean.cuda())
            frames_input = Variable(frames_input.cuda())

            # Evaluate model and optimize it
            x_list = model(frames_input)

            loss = criterion(x_list[-1], frame_clean)
            for i in range(1, len(x_list)-1): # 1, 2, 3, 4
                loss += 0.0001 * criterion(x_list[i], frame_clean)

            losses += loss.item()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                # Results
                model.eval()
                psnr_train = batch_PSNR(x_list[-1], frame_clean, 1.)
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
        save_checkpoint(model, optimizer, epoch + 1, ploty, args.ckp_dir + 'checkpoint_%d_%depoch.ckpt' %
                        (args.sigma, epoch + 1))
        # save denoise.pkl
        torch.save(model, os.path.join(args.save_dir + '/denoising_%d_%d.pkl' % (args.sigma, epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VideoDenoising Training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--loss_pic', type=str, default='loss_pic.png')
    parser.add_argument('--savetxt', type=str, default='loss.txt')
    parser.add_argument('--filelist', type=str, default='./list/train_list.txt')
    parser.add_argument('--save_dir', type=str, default='./mynetwork/')
    parser.add_argument('--pretrained_model', type=str, default='./mynetwork/align.pth')
    parser.add_argument('--ckp_dir', type=str, default='./checkpoints/')
    parser.add_argument('--color', default=True, help='Train with color instead of grayscale')
    parser.add_argument('--use_checkpoint', default=False)
    parser.add_argument('--start_epoch', type=int, default=0, help='Number of start training epoch')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--stages', type=int, default=5, help='Number of iterations')

    ### change with your settings ###
    parser.add_argument('--sigma', type=float, default=25, help='Simulated noise level')
    parser.add_argument('--train_dir', type=str, default='I:/Datasets/vimeo_septuplet/sequences_with_noise_')
    parser.add_argument('--gt_dir', type=str, default='I:/Datasets/vimeo_septuplet/sequences/')

    args = parser.parse_args()

    # modify some parameters
    args.loss_pic = args.loss_pic[:-4] + "_%d.png" % args.sigma
    args.savetxt = args.savetxt[:-4] + "_%d.txt" % args.sigma
    args.train_dir = args.train_dir + '%d/' % args.sigma
    args.pretrained_model = args.pretrained_model[:-4] + '_%d.pth' % args.sigma

    print(args)

    main(args)
