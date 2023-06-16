from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import paddle


class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,\
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):
        interval = self.shuffle_config['interval']
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        return x_student, x_teacher

    def transform(self, x):
        B, C, W, H = x.shape
        loss_type = 'channel'
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = 1
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher, gt=None, n_iter=1):
        if self.warmup_config:
            self.warmup(n_iter)
        if self.earlydecay_config:
            self.earlydecay(n_iter)

        if self.resize_config:
            x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        if self.shuffle_config:
            x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        # if self.transform_config:
        x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)

        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)

        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        loss = self.alpha * loss
        return loss


# class BCELOSS(nn.Module):
#     def __init__(self):
#         super(BCELOSS, self).__init__()
#         self.nll_lose = nn.BCELoss()
#
#     def forward(self, input_scale, taeget_scale):
#         losses = []
#         for inputs, targets in zip(input_scale, taeget_scale):
#             lossall = self.nll_lose(inputs, targets)
#             losses.append(lossall)
#         total_loss = sum(losses)
#         return total_loss
#
#
# class FLoss(nn.Module):
#     def __init__(self):
#         super(FLoss, self).__init__()
#
#         self.conv_mask_s = nn.Conv2d(1, 1, kernel_size=1)
#         self.conv_mask_t = nn.Conv2d(1, 1, kernel_size=1)
#         self.channel_add_conv_s = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=1),
#             nn.LayerNorm([1, 1, 1]),
#             nn.ReLU(inplace=True),  # yapf: disable
#             nn.Conv2d(1, 1, kernel_size=1))
#         self.channel_add_conv_t = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=1),
#             nn.LayerNorm([1, 1, 1]),
#             nn.ReLU(inplace=True),  # yapf: disable
#             nn.Conv2d(1, 1, kernel_size=1))
#
#     def spatial_pool(self, x, in_type):
#         batch, channel, width, height = x.size()
#         input_x = x
#         # [N, C, H * W]
#         input_x = input_x.view(batch, channel, height * width)
#         # [N, 1, C, H * W]
#         input_x = input_x.unsqueeze(1)
#         # [N, 1, H, W]
#         if in_type == 0:
#             context_mask = self.conv_mask_s(x)
#         else:
#             context_mask = self.conv_mask_t(x)
#         # [N, 1, H * W]
#         context_mask = context_mask.view(batch, 1, height * width)
#         # [N, 1, H * W]
#         context_mask = F.softmax(context_mask, dim=2)
#         # [N, 1, H * W, 1]
#         context_mask = context_mask.unsqueeze(-1)
#         # [N, 1, C, 1]
#         context = torch.matmul(input_x, context_mask)
#         # [N, C, 1, 1]
#         context = context.view(batch, channel, 1, 1)
#
#         return context
#
#     def forward(self, preds_S, preds_T):
#         loss_mse = nn.MSELoss(reduction='sum')
#
#         context_s = self.spatial_pool(preds_S, 0)
#         context_t = self.spatial_pool(preds_T, 1)
#
#         out_s = preds_S
#         out_t = preds_T
#
#         channel_add_s = self.channel_add_conv_s(context_s)
#         out_s = out_s + channel_add_s
#
#         channel_add_t = self.channel_add_conv_t(context_t)
#         out_t = out_t + channel_add_t
#
#         rela_loss = loss_mse(out_s, out_t) / len(out_s)
#
#         return rela_loss
#
#
# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T
#
#     def forward(self, y_s, y_t):
#         # print(y_t)
#         p_s = F.sigmoid(y_s)
#         # p_s = F.log_softmax(y_s/self.T, dim=1)
#         # print(p_s)
#         p_t = F.sigmoid(y_t)
#         # p_t = F.softmax(y_t/self.T, dim=1)
#         # print(p_t)
#         loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
#         # print(loss)
#         return loss
#
#
# class diceloss(nn.Module):
#     def __init__(self,):
#         super(diceloss, self).__init__()
#
#     def forward(self, logits, targets):
#         num = targets.size(0)
#         smooth = 1
#
#         probs = torch.sigmoid(logits)
#         targets = torch.sigmoid(targets)
#         m1 = probs.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)
#
#         score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
#         score = 1 - score.sum() / num
#         return score


def dice_loss(pred, mask):
    num = mask.size(0)
    mask = torch.sigmoid(mask / 7)
    pred = torch.sigmoid(pred / 7)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = 1 - dice.sum() / num
    return dice


# class MSELoss(nn.Module):
#     def __init__(self):
#         super(MSELoss, self).__init__()
#         self.loss_fn = nn.MSELoss()
#
#     def forward(self, density_map, gt_data):
#         loss = self.loss_fn(density_map, gt_data)
#         return loss


if __name__ == '__main__':
    rgb = torch.randn(4, 1, 1, 1)
    t = torch.randn(4, 1, 1, 1)
    # net = dice_loss(rgb, t)
    # a = dice_loss(rgb, t)
    net = MSELoss()
    a = net(rgb, t)
    print(a)