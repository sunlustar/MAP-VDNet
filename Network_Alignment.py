import math
import torch

Backward_tensorGrid = {}

def normalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
    tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
    tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


def denormalize(tensorInput):
    tensorRed = (tensorInput[:, 0:1, :, :] * 0.229) + 0.485
    tensorGreen = (tensorInput[:, 1:2, :, :] * 0.224) + 0.456
    tensorBlue = (tensorInput[:, 2:3, :, :] * 0.225) + 0.406
    return torch.cat([tensorRed, tensorGreen, tensorBlue], 1)


def Backward(tensorInput, tensorFlow, cuda_flag):
    # if str(tensorFlow.size()) not in Backward_tensorGrid:
    tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3))\
        .expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
    tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1)\
        .expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        #torch.linspace返回一个一维的tensor（张量），这个张量包含了从start到end，分成steps个线段得到的向量。
    if cuda_flag:
        Backward_tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1).cuda(tensorFlow.device.index)
    else:
        Backward_tensorGrid = torch.cat([tensorHorizontal, tensorVertical], 1)

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    a = Backward_tensorGrid
    b = a + tensorFlow
    grid = b.permute(0, 2, 3, 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=grid, mode='bilinear', padding_mode='border')


class Basic(torch.nn.Module):
    def __init__(self, intLevel):
        super(Basic, self).__init__()
        self.moduleBasic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )
    def forward(self, tensorInput):
        return self.moduleBasic(tensorInput)

class SpyNet(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(SpyNet, self).__init__()
        self.cuda_flag = cuda_flag
        self.moduleBasic = torch.nn.ModuleList([Basic(intLevel) for intLevel in range(4)])

    def forward(self, tensorFirst, tensorSecond):
        tensorFirst = [tensorFirst]
        tensorSecond = [tensorSecond]
        scale = 3 #ori=3
        for intLevel in range(scale):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))

        tensorFlow = tensorFirst[0].new_zeros(tensorFirst[0].size(0), 2,
                                              int(math.floor(tensorFirst[0].size(2) / 2.0)),
                                              int(math.floor(tensorFirst[0].size(3) / 2.0)))

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            # if the sizes of upsampling and downsampling are not the same, apply zero-padding.
            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3):
                tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[0, 1, 0, 0], mode='replicate')
            # input ：[first picture of corresponding level,
            # 		   the output of w with input second picture of corresponding level and upsampling flow,
            # 		   upsampling flow]
            # then we obtain the final flow. 最终再加起来得到intLevel的flow
            tensorFlow = self.moduleBasic[intLevel](torch.cat([tensorFirst[intLevel],
                                                               Backward(tensorInput=tensorSecond[intLevel],
                                                                        tensorFlow=tensorUpsampled,
                                                                        cuda_flag=self.cuda_flag),
                                                               tensorUpsampled], 1)) + tensorUpsampled
        return tensorFlow


class warp(torch.nn.Module):
    def __init__(self):
        super(warp, self).__init__()

    def init_addterm(self, height, width):
        n = torch.FloatTensor(list(range(width)))
        horizontal_term = n.expand((1, 1, height, width))  # 第一个1是batch size
        n = torch.FloatTensor(list(range(height)))
        vertical_term = n.expand((1, 1, width, height)).permute(0, 1, 3, 2)
        addterm = torch.cat((horizontal_term, vertical_term), dim=1)
        return addterm

    def forward(self, frame, flow):
        """
        :param frame: frame.shape (batch_size=1, n_channels=3, width=256, height=448)
        :param flow: flow.shape (batch_size=1, n_channels=2, width=256, height=448)
        :return: reference_frame: warped frame
        """
        b, c, h, w = frame.size()

        addterm = self.init_addterm(h, w)

        flow = flow + addterm.cuda()

        horizontal_flow = torch.zeros(b, 1, frame.size(2), frame.size(3)).cuda()
        vertical_flow = torch.zeros(b, 1, frame.size(2), frame.size(3)).cuda()
        for i in range(b):
            horizontal_flow[i, :, :, :] = flow[i, 0, :, :].expand(1, 1, h, w)
            vertical_flow[i, :, :, :] = flow[i, 1, :, :].expand(1, 1, h, w)

        horizontal_flow = horizontal_flow * 2 / (w - 1) - 1
        vertical_flow = vertical_flow * 2 / (h - 1) - 1
        flow = torch.cat((horizontal_flow, vertical_flow), dim=1)
        flow = flow.permute(0, 2, 3, 1)
        reference_frame = torch.nn.functional.grid_sample(frame, flow)
        return reference_frame



class Alignment(torch.nn.Module):
    def __init__(self, cuda_flag):
        super(Alignment, self).__init__()
        self.cuda_flag = cuda_flag

        self.SpyNet = SpyNet(cuda_flag=self.cuda_flag)
        self.warp = warp()

    # frames should be TensorFloat
    def forward(self, frames):
        """
        :param frames: [batch_size=1, img_num, n_channels=3, h, w]
        :return:
        """
        # noise train _all
        for i in range(frames.size(1)):
            frames[:, i, :, :, :] = normalize(frames[:, i, :, :, :])

        if self.cuda_flag:
            opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4)).cuda()
            warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4)).cuda()
        else:
            opticalflows = torch.zeros(frames.size(0), frames.size(1), 2, frames.size(3), frames.size(4))
            warpframes = torch.empty(frames.size(0), frames.size(1), 3, frames.size(3), frames.size(4))


        process_index = [0, 1, 2, 4, 5, 6]
        for i in process_index:
            opticalflows[:, i, :, :, :] = self.SpyNet(frames[:, 3, :, :, :], frames[:, i, :, :, :])
        warpframes[:, 3, :, :, :] = denormalize(frames[:, 3, :, :, :])

        for i in process_index:
            warpframes[:, i, :, :, :] = denormalize(self.warp(frames[:, i, :, :, :], opticalflows[:, i, :, :, :]))
        return warpframes