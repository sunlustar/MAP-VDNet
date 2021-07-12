import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn
from Network_Alignment import Alignment


class Encodingblk(nn.Module):
    def __init__(self, c_in):
        super(Encodingblk, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.down = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        out = F.relu(self.conv1(input))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        f_e = self.conv4(out)
        down = F.relu(self.down(f_e))
        return f_e, down


class Encodingblk_end(nn.Module):
    def __init__(self, c_in):
        super(Encodingblk_end, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

    def forward(self, input):
        out = F.relu(self.conv1(input))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        f_e = self.conv4(out)
        return f_e


class Decodingblk(nn.Module):
    def __init__(self, out_channels):
        super(Decodingblk, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

        self.up = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)

    def forward(self, input, map):
        up = self.up(input)
        cat = torch.cat((up, map), dim=1)
        cat = F.relu(self.conv1(cat))
        out = F.relu(self.conv2(cat))
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        return out

class ResBlock(torch.nn.Module):
    def __init__(self, layer):
        super(ResBlock, self).__init__()
        self.layer = layer

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        out = x + input
        return out

class Filters(torch.nn.Module):
    def __init__(self, channel):
        super(Filters, self).__init__()

        self.channel = channel

        self.conv1 = nn.Conv2d(in_channels=3 * 2, out_channels=self.channel, kernel_size=3, padding=1)
        self.resnet = nn.Sequential(
            ResBlock(1),
            ResBlock(2),
            ResBlock(3),
            ResBlock(4),
            ResBlock(5),
        )
        self.conv2 = nn.Conv2d(in_channels=self.channel, out_channels=3, kernel_size=3, padding=1)

    def forward(self, frames):  ##### [b,n,c,h,w] ; burst_length=8 ; h=w=128 ; channel=3
        filters = torch.zeros(frames.size(0), frames.size(1), frames.size(2), frames.size(3), frames.size(4)).cuda()

        for i in range(frames.size(1)):
            x_cat = torch.cat((frames[:, i, :, :, :], frames[:, 3, :, :, :]), dim=1)
            fea = F.relu(self.conv1(x_cat))
            outfea = self.resnet(fea)
            filters[:, i, :, :, :] = self.conv2(outfea)  # [b,3,h,w]

        filters = F.softmax(filters, dim=1)
        outframes = torch.mul(frames, filters)  # [b,n,3,h,w]
        output = torch.sum(outframes, dim=1)
        return output


class MAPVDNet(nn.Module):
    def __init__(self, cuda_flag=True, alignment_model=None, T=5, feature_maps=64, kSize=3):
        super(MAPVDNet, self).__init__()

        self.cuda_flag = cuda_flag

        self.Alignment = Alignment(cuda_flag=self.cuda_flag)
        if alignment_model is not None:
            pretrained_dict = torch.load(alignment_model)
            self.Alignment.load_state_dict(pretrained_dict)

        self.Filters = Filters(32)

        self.Encoding_block1 = Encodingblk(feature_maps)
        self.Encoding_block2 = Encodingblk(feature_maps)
        self.Encoding_block3 = Encodingblk(feature_maps)
        self.Encoding_block4 = Encodingblk(feature_maps)

        self.Encoding_block_end = Encodingblk_end(feature_maps)

        self.Decoding_block1 = Decodingblk(feature_maps)
        self.Decoding_block2 = Decodingblk(feature_maps)
        self.Decoding_block3 = Decodingblk(feature_maps)

        self.feature_decoding_end = Decodingblk(feature_maps)

        self.Fe_e = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d(3, feature_maps, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(feature_maps, feature_maps, kSize, padding=(kSize - 1) // 2, stride=1)
        ]) for _ in range(T)])

        self.Fe_f = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d((i + 2) * feature_maps, feature_maps, 1, padding=0, stride=1)]) for i in range(T - 1)])

        self.construction = nn.Conv2d(feature_maps, kSize, kSize, padding=1)
        self.constructions = nn.ModuleList([nn.Sequential(*[
            nn.Conv2d((i + 2) * feature_maps, feature_maps, 1, padding=0, stride=1),
            nn.Conv2d(feature_maps, feature_maps, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(feature_maps, kSize, kSize, padding=1)
        ]) for i in range(T-1)])

        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.8)) for _ in range(T)])

    def recon(self, v, ave_neighboors, id_layer):
        ETA = self.eta[id_layer]
        out = (ave_neighboors + ETA * v) / (1 + ETA)
        return out

    def forward(self, input):
        ### Alignment module ###
        warpframes = self.Alignment(input)

        ### Spatial adaptive fusion ###
        ave_neighboors = self.Filters(warpframes)

        x_list = []
        x = ave_neighboors
        x_list.append(x)

        fea_list = []
        V_list = []

        ### U-net denoiser ###
        for i in range(len(self.Fe_e)):
            fea = self.Fe_e[i](x)
            fea_list.append(fea)
            if i != 0:
                fea = self.Fe_f[i - 1](torch.cat(fea_list, 1))

            encode0, down0 = self.Encoding_block1(fea)
            encode1, down1 = self.Encoding_block2(down0)
            encode2, down2 = self.Encoding_block3(down1)
            encode3, down3 = self.Encoding_block4(down2)

            media_end = self.Encoding_block_end(down3)

            decode3 = self.Decoding_block1(media_end, encode3)
            decode2 = self.Decoding_block2(decode3, encode2)
            decode1 = self.Decoding_block3(decode2, encode1)
            decode0 = self.feature_decoding_end(decode1, encode0)

            V_list.append(decode0)

            if i == 0:
                decode0 = self.construction(F.relu(decode0))
            else:
                decode0 = self.constructions[i - 1](torch.cat(V_list, 1))

            conv_out = x + decode0

            x = self.recon(conv_out, ave_neighboors, id_layer=i) ### Reconstruction
            x_list.append(x)
        return x_list


if __name__ == '__main__':
    net = MAPVDNet().cuda()
    # print(net)
    x = Variable(torch.FloatTensor(1, 7, 3, 64, 64)).cuda()
    y = net(x)
    print(y.data.shape)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)