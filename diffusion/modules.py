# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        channels,
        size,
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        residual=False,
    ):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_dim=256,
    ):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class FmriEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        time_dim=256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.fc1 = nn.Linear(input_dim, time_dim)
        self.bn1 = nn.BatchNorm1d(time_dim)
        self.fc2 = nn.Linear(time_dim, time_dim)
        self.bn2 = nn.BatchNorm1d(time_dim)
        self.fc3 = nn.Linear(time_dim, time_dim)
        self.bn3 = nn.BatchNorm1d(time_dim)

    def forward(self, x):
        x = F.gelu(self.bn1(self.fc1(x)))
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x


class FmriRidgeEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        time_dim=256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.fc = nn.Linear(input_dim, time_dim, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x

    def l2_regularization(self, alpha=1):
        return alpha * torch.norm(self.fc.weight, p=2)


class FmriNormEncoder(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        time_dim=256,
        scale=1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.fc = nn.utils.weight_norm(
            nn.Linear(input_dim, time_dim, bias=False),
            name="weight",
            dim=0,
        )
        self.scale = nn.Parameter(torch.Tensor([scale]))

    def forward(self, x):
        x = self.scale * self.fc(x)
        return x


class UNet_conditional(nn.Module):
    def __init__(
        self,
        c_in=3,
        c_out=3,
        time_dim=256,
        mri_dim=2,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        self.fmri_emb = FmriEncoder(mri_dim, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, channels, 2, device=self.device).float()
                / channels
            )
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        if y is not None:
            t += self.fmri_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == "__main__":
    net = UNet_conditional(mri_dim=2, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(30, 3, 64, 64, dtype=torch.float32)
    t = x.new_tensor([500] * x.shape[0]).long()
    y = x.new_tensor(torch.randn(x.shape[0], 1024))
    print(net(x, t, y).shape)
