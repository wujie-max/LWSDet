import torch
import torch.nn as nn
from torch.functional import F
from functools import partial

from ultralytics.nn.modules.conv import autopad


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



class PGConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, s, padding, groups, dilation, activation)."""

    # default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k, s, act=True, n_div=4):
        super().__init__()
        self.dim_conv3 = c1 // n_div
        self.dim_untouched = c1 - self.dim_conv3
        # self.c = int(c2 * 0.5)
        self.partial_conv3 = Conv(self.dim_conv3, self.dim_conv3, k, s, k//2)
        self.dwconv = nn.Sequential(
                nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, k//2, groups=self.dim_conv3, bias=False),
                # nn.AvgPool2d(3, 1, 1),
                nn.BatchNorm2d(self.dim_conv3),
                nn.SiLU(inplace=True) if act else nn.Sequential(),
            )
        # self.dwconv = nn.Conv2d(self.dim_conv3, self.dim_conv3, k, 1, k//2, groups=self.dim_conv3, bias=False)
        # self.cv1 = nn.Conv2d(self.dim_untouched, self.c, 3, s, 1, groups=math.gcd(self.dim_untouched, self.c), bias=False)
        # self.avg = nn.AvgPool2d(s, s, 0)
        self.avg = nn.AvgPool2d(s, s, 0)
        self.bn = nn.BatchNorm2d(self.dim_untouched + 2 * self.dim_conv3)
        self.act = nn.SiLU()
        self.cv2 = Conv(self.dim_untouched + 2 * self.dim_conv3, c2, 1, 1, 0)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x1 = torch.cat((x1, self.dwconv(x1)), 1)
        x2 = self.act(self.bn(torch.cat((self.avg(x2), x1), 1)))

        return self.cv2(x2)

class MLFF(nn.Module):
    def __init__(self, channel, out_channels):
        super().__init__()
        # input = channel[0]+channel[1]+channel[2]
        input = channel[0]+2*128
        self.silu = nn.SiLU()
        self.conv = Conv(input, out_channels, 1, 1, 0)
        self.conv1 = Conv(channel[1], 128, 1, 1, 0)
        self.conv2 = Conv(channel[2], 128, 1, 1, 0)

    def forward(self, xs):
        target_size = xs[0].shape[-2:]
        xs[1] = self.conv1(xs[1])
        xs[2] = self.conv2(xs[2])
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[0]:
                xs[i] = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                xs[i] = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
        return self.conv(torch.cat((xs[0], xs[1], xs[2]), 1))


# GCLv1
class AFSBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)

        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        # self.fc2 = nn.Linear(hidden, dim)
        self.fc2 = nn.Conv2d(hidden, dim, 1, 1)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x  # [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c).permute(0, 2, 3, 1)
        # c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2((self.act(g) * torch.cat((i, c), dim=-1)).permute(0, 3, 1, 2))
        x = self.drop_path(x)
        return x + shortcut


class AFSModule(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, ratio=2):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        # self.c = int(c2 * 0.8)  # hidden channels
        # self.ls = LSBlock(c1, c2)
        m = nn.ModuleList(AFSBlock(c2, expansion_ratio=ratio) for _ in range(n))
        # self.conv2 = Conv(self.c, c2, 1, 1)
        self.GCL = nn.Sequential(*m)
        # self.fg = FGlo(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        # x = self.ls(x)
        return self.GCL(x)
        # return self.conv2(self.GCL(self.conv1(x)))+x