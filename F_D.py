import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DeMU(nn.Module):
    def __init__(self, channel):
        super(DeMU, self).__init__()

        self.compare = CompareW()
        self.cconv = nn.Conv2d(channel, channel, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, t):
        w3, w3_3 = self.compare(rgb, t)
        if w3 >= w3_3:
            rgb = rgb
            t = t
        else:
            rgb = t
            t = rgb

        redc = self.cconv(rgb + t)
        gate = torch.sigmoid(-redc)
        rgb = self.cconv(rgb)
        concat2 = self.conv(rgb * gate + rgb)
        return concat2


class QKVF(nn.Module):
    def __init__(self, channel, drop_path=0.2):
        super(QKVF, self).__init__()
        self.pah = OverlapPatchEmbed(3, channel)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(channel, channel, head_count=4, value_channels=channel)
        self.norm = nn.LayerNorm(channel)
        self.co = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.relu = nn.ReLU()

    def forward(self, x, y):
        b = x.shape[0]

        resar1, h, w = self.pah(x)
        resat1, h, w = self.pah(y)

        fina1 = resar1 + self.drop_path(self.attn(self.norm(resar1), self.norm(resat1)))
        fina2 = resat1 + self.drop_path(self.attn(self.norm(resat1), self.norm(resar1)))

        fin1 = self.norm(fina1)
        fin1 = fin1.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        fin1 = fin1 * x + x
        fin2 = self.norm(fina2)
        fin2 = fin2.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        fin2 = fin2 * y + y
        ft = self.co(torch.cat((fin1 * fin2, fin1 + fin2), 1))
        fin1 = self.relu(fin1)
        fin2 = self.relu(fin2)

        return fin1, fin2, ft


class GCN(nn.Module):
    def __init__(self, planes, ratio=4):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes // ratio * 2)
        self.conv2 = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes // ratio)

        self.conv1d1 = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn1d1 = nn.BatchNorm1d(planes // ratio)

        self.conv1d2 = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn1d2 = nn.BatchNorm1d(planes // ratio * 2)

        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.finalr = nn.ReLU()
        self.finalt = nn.ReLU()
        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(planes),
                                   nn.ReLU()
                                   )

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, x, y):
        x_s = self.conv1(x)
        x_s = self.bn1(x_s)
        x_s = self.to_matrix(x_s)
        y_s = self.conv1(y)
        y_s = self.bn1(y_s)
        y_s = self.to_matrix(y_s)

        b = self.conv2(y)
        b = self.bn2(b)
        b = self.to_matrix(b)
        c = self.conv2(x)
        c = self.bn2(c)
        c = self.to_matrix(c)

        z_i = torch.matmul(x_s, b.transpose(1, 2))
        z_iy = torch.matmul(y_s, c.transpose(1, 2))

        z = z_i.transpose(1, 2).contiguous()
        zy = z_iy.transpose(1, 2).contiguous()

        z = self.conv1d1(z)
        z = self.bn1d1(z)
        zy = self.conv1d1(zy)
        zy = self.bn1d1(zy)

        z = z.transpose(1, 2).contiguous()
        z += z_i
        zy = zy.transpose(1, 2).contiguous()
        zy += z_iy

        z = self.conv1d2(z)
        z = self.bn1d2(z)
        zy = self.conv1d2(zy)
        zy = self.bn1d2(zy)

        v = torch.matmul(z, b)
        vy = torch.matmul(zy, c)

        n, _, h, w = x.size()
        v = v.view(n, -1, h, w)
        vy = vy.view(n, -1, h, w)

        v = self.conv3(v)
        v = self.bn3(v)
        vy = self.conv3(vy)
        vy = self.bn3(vy)
        finr = self.finalr(x + v)
        fint = self.finalt(y + vy)
        fin = self.final(torch.cat((finr + fint, finr * fint), 1))

        return fin


class EhLow(nn.Module):
    def __init__(self, channels):
        super(EhLow, self).__init__()

        self.conv1_1 = nn.Conv2d(channels // 2, channels // 2, kernel_size=1)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(True)
        )
        # self.conv3_3 = nn.Sequential(
        #     nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channels),
        #     nn.ReLU(True)
        # )
        self.cconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, total, high):
        infer, infeer = total.chunk(2, 1)
        infe, infee = high.chunk(2, 1)

        cat1 = self.sigmoid(self.conv1_1(infer - infe))
        cob1 = self.conv2_2(infer * cat1 + infer + infe)

        cat2 = self.sigmoid(self.conv1_1(infeer - infee))
        cob2 = self.conv2_2(infeer * cat2 + infeer + infee)

        jh = self.cconv(torch.cat((cob1, cob2), 1))
        # jhres = self.cconv(jh + high + total)

        return jh


class EhHigh(nn.Module):
    def __init__(self, channels):
        super(EhHigh, self).__init__()

        self.conv1_1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True)
        )
        self.cconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, total, high):
        cat1 = self.softmax(total * high)
        mul1 = self.conv1_1(total)
        cob1 = self.conv2_2(mul1 * cat1)

        jh = self.cconv(total + cob1)

        return jh


class CompareW(nn.Module):
    def __init__(self):
        super(CompareW, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        midx = self.sigmoid(x)
        midy = self.sigmoid(y)
        # tsdx = midx.std()
        # tsdy = midy.std()
        finx = midx >= 0.5
        finy = midy >= 0.5
        a, b = torch.nonzero(finx).shape
        c, d = torch.nonzero(finy).shape

        return a, c


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)
        q = self.norm(q)
        return U * q


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)
        z = self.Conv_Squeeze(z)
        z = self.Conv_Excitation(z)
        z = self.norm(z)
        return U * z.expand_as(U)


class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse + U_sse


class OverlapPatchEmbed(nn.Module):

    def __init__(self, patch_size, channel):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(channel, channel, kernel_size=patch_size, stride=1, padding=1)
        self.norm = nn.LayerNorm(channel)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Attention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels, bias=True)
        self.queries = nn.Linear(in_channels, key_channels, bias=True)
        self.values = nn.Linear(in_channels, key_channels, bias=True)
        self.reprojection = nn.Linear(in_channels, key_channels)

    def forward(self, input_, y):
        keys = self.keys(input_ + y).permute(0, 2, 1)
        queries = self.queries(y).permute(0, 2, 1)
        values = self.values(input_).permute(0, 2, 1)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                            :,
                            i * head_key_channels: (i + 1) * head_key_channels,
                            :
                            ], dim=2)
            query = F.softmax(queries[
                              :,
                              i * head_key_channels: (i + 1) * head_key_channels,
                              :
                              ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]

            context = key @ value.transpose(1, 2)

            attended_value = context.transpose(1, 2) @ query
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1, 2)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value