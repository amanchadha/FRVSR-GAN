"""
This file contains implementation of FRVSR (FNet and SRNet) from https://arxiv.org/abs/1801.04590
aman@amanchadha.com

Adapted from FR-SRGAN, MIT 6.819 Advances in Computer Vision, Nov 2018
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import vgg16

class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=conv_dim, out_channels=conv_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = func.relu(out)
        out = self.conv2(out)
        out = input + out
        return out


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                               kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.conv1(input)
        out = func.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = func.leaky_relu(out, 0.2)
        return out


class FNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, typ):
        super(FNetBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if typ == "maxpool":
            self.final = lambda x: func.max_pool2d(x, kernel_size=2)
        elif typ == "bilinear":
            self.final = lambda x: func.interpolate(x, scale_factor=2, mode="bilinear")
        else:
            raise Exception('Type does not match any of maxpool or bilinear')

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out


class SRNet(nn.Module):
    def __init__(self, in_dim=51):
        super(SRNet, self).__init__()
        self.inputConv = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.ResBlocks = nn.Sequential(*[ResBlock(64) for i in range(10)])
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        self.outputConv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        # self.dropout = nn.Dropout(p = 0.5)

    def forward(self, input):
        out = self.inputConv(input)
        out = self.ResBlocks(out)
        out = self.deconv1(out)
        out = func.relu(out)
        out = self.deconv2(out)
        out = func.relu(out)
        out = self.outputConv(out)
        return out


class FNet(nn.Module):
    def __init__(self, in_dim=6):
        super(FNet, self).__init__()
        self.convPool1 = FNetBlock(in_dim, 32, typ="maxpool")
        self.convPool2 = FNetBlock(32, 64, typ="maxpool")
        self.convPool3 = FNetBlock(64, 128, typ="maxpool")
        self.convBinl1 = FNetBlock(128, 256, typ="bilinear")
        self.convBinl2 = FNetBlock(256, 128, typ="bilinear")
        self.convBinl3 = FNetBlock(128, 64, typ="bilinear")
        self.seq = nn.Sequential(self.convPool1, self.convPool2, self.convPool3,
                                 self.convBinl1, self.convBinl2, self.convBinl3)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        out = self.seq(input)
        out = self.conv1(out)
        out = func.leaky_relu(out, 0.2)
        out = self.conv2(out)
        self.out = torch.tanh(out)
        self.out.retain_grad()
        return self.out


# please ensure that input is (batch_size, depth, height, width)
# courtesy to Hung Nguyen at https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec.
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

# please ensure that lr_height and lr_width must be a multiple of 8.
class FRVSR(nn.Module):
    def __init__(self, batch_size, lr_height, lr_width):
        super(FRVSR, self).__init__()
        FRVSR.SRFactor = 4
        self.width = lr_width
        self.height = lr_height
        self.batch_size = batch_size
        self.fnet = FNet()
        self.todepth = SpaceToDepth(FRVSR.SRFactor)
        self.srnet = SRNet(FRVSR.SRFactor * FRVSR.SRFactor * 3 + 3)  # 3 is channel number

    # make sure to call this before every batch train.
    def init_hidden(self, device):
        self.lastLrImg = torch.zeros([self.batch_size, 3, self.height, self.width]).to(device)
        self.EstHrImg = torch.zeros([self.batch_size, 3, self.height * FRVSR.SRFactor, self.width * FRVSR.SRFactor]).to(device)
        height_gap = 2 / (self.height - 1)
        width_gap = 2 / (self.width - 1)
        height, width = torch.meshgrid([torch.range(-1, 1, height_gap), torch.range(-1, 1, width_gap)])
        self.lr_identity = torch.stack([width, height]).to(device)

        height_gap = 2 / (self.height * self.SRFactor - 1)
        width_gap = 2 / (self.width * self.SRFactor - 1)
        height, width = torch.meshgrid([torch.range(-1, 1, height_gap), torch.range(-1, 1, width_gap)])
        self.hr_identity = torch.stack([width, height]).to(device)

    # x is a 4-d tensor of shape N×C×H×W
    def forward(self, input):
        # Apply FNet
        # print(f'input.shape is {input.shape}, lastImg shape is {self.lastLrImg.shape}')
        preflow = torch.cat((input, self.lastLrImg), dim=1)
        flow = self.fnet(preflow)
        relative_place = flow + self.lr_identity
        self.EstLrImg = func.grid_sample(self.lastLrImg, relative_place.permute(0, 2, 3, 1))
        # print(self.EstLrImg)
        relative_placeNCHW = func.interpolate(relative_place, scale_factor=4, mode="bilinear")
        relative_placeNWHC = relative_placeNCHW.permute(0, 2, 3, 1)  # switch to channels-last notation for grid_sample()
        afterWarp = func.grid_sample(self.EstHrImg, relative_placeNWHC)
        self.afterWarp = afterWarp  # for debugging, should be removed later.
        depthImg = self.todepth(afterWarp)

        # Apply SRNet
        srInput = torch.cat((input, depthImg), dim=1)
        estImg = self.srnet(srInput)
        self.lastLrImg = input
        self.EstHrImg = estImg
        self.EstHrImg.retain_grad()
        return self.EstHrImg, self.EstLrImg

    def set_param(self, **kwargs):
        for (key, val) in kwargs.items():
            if key == 'batch_size':
                self.batch_size = val
            if key == 'height':
                self.height = val
            if key == 'width':
                self.width = val

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # Adversarial Loss
        # adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, hr_est, hr_img, lr_est, lr_img, idx):
        # Adversarial Loss
        adversarial_loss = -torch.mean(out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(hr_est), self.loss_network(hr_img))
        # Image Loss
        image_loss = self.mse_loss(hr_est, hr_img)
        # TV Loss
        tv_loss = self.tv_loss(hr_est)
        # flow loss
        if idx != 0:
            flow_loss = self.mse_loss(lr_est, lr_img)
        else:
            flow_loss = 0

        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss + 0.0001 * flow_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


#
# if __name__ == "__main__":
#     g_loss = GeneratorLoss()
#     print(g_loss)

# class FRVSR_Criterion(torch.autograd.Function):
#     def __init__(self):
#         super(FRVSR_Criterion, self).__init__()
#
#     def forward(self, lr_est, lr_img, hr_est, hr_img):
#         #= input[0], input[1], input[2], input[3]
#         assert (lr_est.shape == lr_img.shape)
#         assert (hr_est.shape == hr_img.shape)
#         return nn.MSELoss(lr_est, lr_img) + nn.MSELoss(hr_est, hr_img)

# run tests make sure that output is correct.
class TestFRVSR(unittest.TestCase):
    def testResBlock(self):
        block = ResBlock(3)
        input = torch.rand(2, 3, 64, 112)
        output = block(input)
        self.assertEqual(input.shape, output.shape)

    def testConvLeaky(self):
        block = ConvLeaky(3, 32)
        input = torch.rand(2, 3, 64, 112)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 32, 64, 112).shape)

    def testFNetBlockMaxPool(self):
        block = FNetBlock(3, 32, "maxpool")
        input = torch.rand(2, 3, 64, 112)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 32, 32, 56).shape)

    def testFNetBlockInterPolate(self):
        block = FNetBlock(3, 32, "bilinear")
        input = torch.rand(2, 3, 32, 56)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 32, 64, 112).shape)

    def testSRNet(self):
        block = SRNet()
        input = torch.rand(2, 51, 32, 56)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 3, 128, 224).shape)
        block = SRNet()
        input = torch.rand(2, 51, 64, 64)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 3, 256, 256).shape)

    def testFNet(self):
        block = FNet()
        input = torch.rand(2, 6, 32, 56)
        output = block(input)
        self.assertEqual(output.shape, torch.empty(2, 2, 32, 56).shape)

    def testFRVSR(self):
        H = 16
        W = 16
        block = FRVSR(4, H, W)
        input = torch.rand(7, 4, 3, H, W)
        block.init_hidden(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        for batch_frames in input:
            output1, output2 = block(batch_frames)
            self.assertEqual(output1.shape, torch.empty(4, 3, H * 4, W * 4).shape)
            self.assertEqual(output2.shape, torch.empty(4, 3, H, W).shape)

    # def testCriterion(self):
    #     H = 16
    #     W = 16
    #     input = torch.rand(7, 4, 3, H, W)
    #     output = torch.rand(4, 3, H * 4, W * 4)
    #     criterion = FRVSR_Criterion()
    #     self.assertIsInstance(criterion(input, input, output, output), type(0.1))


if __name__ == '__main__':
    unittest.main()
