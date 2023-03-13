import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from ThreeM.A8.TF9 import QKVF, EhHigh, EhLow, GCN, scSE, DeMU
from Three.Try6.convnext import convnext_small, LayerNorm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg_r = convnext_small(True)
        self.vgg_t = convnext_small(True)

        self.layer0_r = nn.Sequential(self.vgg_r.downsample_layers[0], self.vgg_r.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_r = nn.Sequential(self.vgg_r.downsample_layers[1], self.vgg_r.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_r = nn.Sequential(self.vgg_r.downsample_layers[2], self.vgg_r.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_r = nn.Sequential(self.vgg_r.downsample_layers[3], self.vgg_r.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.layer0_t = nn.Sequential(self.vgg_t.downsample_layers[0], self.vgg_t.stages[0],
                                      LayerNorm(96, eps=1e-6, data_format="channels_first"))
        self.layer1_t = nn.Sequential(self.vgg_t.downsample_layers[1], self.vgg_t.stages[1],
                                      LayerNorm(192, eps=1e-6, data_format="channels_first"))
        self.layer2_t = nn.Sequential(self.vgg_t.downsample_layers[2], self.vgg_t.stages[2],
                                      LayerNorm(384, eps=1e-6, data_format="channels_first"))
        self.layer3_t = nn.Sequential(self.vgg_t.downsample_layers[3], self.vgg_t.stages[3],
                                      LayerNorm(768, eps=1e-6, data_format="channels_first"))

        self.qkv1 = QKVF(96)
        self.qkv2 = QKVF(192)
        self.qkv3 = QKVF(384)
        self.qkv4 = QKVF(768)

        self.gcn = GCN(768)

        self.enhance1 = EhLow(96)
        self.enhance2 = EhLow(192)
        self.enhance3 = EhLow(384)
        self.enhance4 = EhHigh(768)

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(768, 384, 3, 1, 1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(384, 384, 3, 1, 1),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(),
        # )
        self.conv2 = nn.Sequential(
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.conv0 = nn.Sequential(
            nn.Conv2d(96, 96, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.conv0_0 = nn.Sequential(
            nn.Conv2d(48, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.sce4 = scSE(384)
        self.sce3 = scSE(192)
        self.sce2 = scSE(96)
        self.sce1 = scSE(48)

        self.fuf3 = DeMU(384)
        self.fuf2 = DeMU(192)
        self.fuf1 = DeMU(96)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(48, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.reg_layer3 = nn.Sequential(
            nn.Conv2d(192, 1, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.reg_layer2 = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.bianxingconv = nn.Sequential(
            nn.Conv2d(768, 1, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, RGBT):
        image = RGBT[0]
        t = RGBT[1]
        conv1_r = self.layer0_r(image)
        conv1_t = self.layer0_t(t)
        x0_r, x0_t, ft0 = self.qkv1(conv1_r, conv1_t)

        conv2_r = self.layer1_r(x0_r)
        conv2_t = self.layer1_t(x0_t)
        x1_r, x1_t, ft1 = self.qkv2(conv2_r, conv2_t)

        conv3_r = self.layer2_r(x1_r)
        conv3_t = self.layer2_t(x1_t)
        x2_r, x2_t, ft2 = self.qkv3(conv3_r, conv3_t)

        conv4_r = self.layer3_r(x2_r)
        conv4_t = self.layer3_t(x2_t)
        x3_r, x3_t, ft3 = self.qkv4(conv4_r, conv4_t)

        resout = self.gcn(x3_r, x3_t)

        efu4 = self.enhance4(ft3, resout)
        efu3 = self.enhance3(ft2, efu4)
        efu2 = self.enhance2(ft1, efu3)
        efu1 = self.enhance1(ft0, efu2)

        unf1 = self.fuf1(x0_r, x0_t)
        unf2 = self.fuf2(x1_r, x1_t)
        unf3 = self.fuf3(x2_r, x2_t)

        # fin3 = self.conv3(resout)
        fin2 = self.conv2(self.sce4(efu4 * unf3 + efu4))
        fin1 = self.conv1(self.sce3(efu3 * unf2 + efu3 + fin2))
        fin0 = self.conv0(self.sce2(efu2 * unf1 + efu2 + fin1))
        fin = self.conv0_0(self.sce1(efu1 + fin0))

        fin = self.reg_layer(fin)
        out3 = self.reg_layer3(fin2)
        out2 = self.reg_layer2(fin1)
        out1 = self.reg_layer(fin0)

        return fin, out1, out2, out3


if __name__ == '__main__':
    rgb = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 3, 480, 640)
    # rgb = torch.randn(1, 3, 256, 256)
    # depth = torch.randn(1, 3, 256, 256)
    model = Net()
    res = model([rgb, depth])
    # print(res.shape)
    print(res[0].shape, res[1].shape, res[2].shape, res[3].shape)
    # from models.ModelTwo.One.canshu.utils import compute_speed
    # from ptflops import get_model_complexity_info
    #
    # with torch.cuda.device(0):
    #     net = Net()
    #     flops, params = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    #     print('Flops:' + flops)
    #     print('Params:' + params)
    #
    # compute_speed(net, input_size=(1, 3, 480, 640), iteration=500)
