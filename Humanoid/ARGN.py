
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def apply_rope(x):
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype


    theta = 1.0 / (10000 ** (torch.arange(0, C // 2, dtype=dtype, device=device) / (C // 2)))


    pos_h = torch.arange(H, dtype=dtype, device=device).view(1, H, 1, 1)  # (1, H, 1, 1)
    pos_w = torch.arange(W, dtype=dtype, device=device).view(1, 1, W, 1)  # (1, 1, W, 1)


    pos_emb = pos_h * theta.view(1, 1, 1, C // 2) + pos_w * theta.view(1, 1, 1, C // 2)


    x_rot = x.view(B, C // 2, 2, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, C//2, 2)


    cos = torch.cos(pos_emb)  # (1, H, W, C//2)
    sin = torch.sin(pos_emb)  # (1, H, W, C//2)


    x_real = x_rot[..., 0]
    x_imag = x_rot[..., 1]

    x_real_rot = x_real * cos - x_imag * sin
    x_imag_rot = x_real * sin + x_imag * cos


    x_rotated = torch.stack([x_real_rot, x_imag_rot], dim=-1)  # (B, H, W, C//2, 2)
    x_rotated = x_rotated.permute(0, 3, 4, 1, 2).reshape(B, C, H, W)

    return x_rotated

class RWKV_ImageBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, expansion=4):
        super().__init__()

        self.r = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)


        self.w_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1)
        )
        self.u = nn.Parameter(torch.randn(dim))


        hidden_dim = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, hidden=None):
        B, C, H, W = x.shape



        w = self.w_conv(x)  # (B, C, H, W)
        w = w.view(B, C, -1)  # (B, C, H*W)


        x_rope = apply_rope(x)
        r = torch.sigmoid(self.r(x))
        k = self.k(x_rope)
        v = self.v(x_rope)


        k = k.view(B, C, -1)  # (B, C, N)
        v = v.view(B, C, -1)


        if hidden is None:
            numerator = torch.zeros_like(k)
            denominator = torch.zeros_like(k)
        else:
            numerator, denominator = hidden


        new_numerator = numerator * torch.exp(-w) + k * v
        new_denominator = denominator * torch.exp(-w) + k


        wkv = (new_numerator + torch.exp(self.u.view(1, -1, 1)) * k * v) / \
              (new_denominator + torch.exp(self.u.view(1, -1, 1)) * k)


        wkv = wkv.view(B, C, H, W)
        wkv = r * wkv


        residual = x
        x = residual + wkv
        x = x + self.fc2(F.relu(self.fc1(x)) ** 2 )

        return x, (new_numerator.detach(), new_denominator.detach())


class FPN(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for ch in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(ch, out_channels, 1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

    def forward(self, features):

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]


        merged = []
        prev = None
        for lateral in reversed(laterals):
            if prev is not None:
                lateral += F.interpolate(prev, scale_factor=2, mode='nearest')
            prev = lateral
            merged.insert(0, lateral)


        return [self.output_convs[i](feat) for i, feat in enumerate(merged)]


class ARGN_ImageModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )


        self.stage1 = nn.ModuleList([RWKV_ImageBlock(64) for _ in range(2)])
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.stage2 = nn.ModuleList([RWKV_ImageBlock(128) for _ in range(2)])
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.stage3 = nn.ModuleList([RWKV_ImageBlock(256) for _ in range(2)])


        self.fpn = FPN([64, 128, 256], 256)
        self.ead = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.stem(x)


        hidden1 = [None] * len(self.stage1)
        for i, block in enumerate(self.stage1):
            x, hidden1[i] = block(x)
        f1 = x


        x = self.down1(f1)


        hidden2 = [None] * len(self.stage2)
        for i, block in enumerate(self.stage2):
            x, hidden2[i] = block(x)
        f2 = x


        x = self.down2(f2)


        hidden3 = [None] * len(self.stage3)
        for i, block in enumerate(self.stage3):
            x, hidden3[i] = block(x)
        f3 = x


        features = self.fpn([f1, f2, f3])
        fused = sum([F.adaptive_avg_pool2d(feat, features[0].shape[-2:])
                     for feat in features])

        return self.head(fused)


model = ARGN_ImageModel(num_classes=6)
x = torch.rand(2, 3, 224, 224)
y = model(x)
print(y.shape)