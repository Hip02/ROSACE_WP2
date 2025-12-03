import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)


class WeakCMEUNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = UNetBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = UNetBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = UNetBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32)

        # Final segmentation map
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoding
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))

        # Decoding
        y3 = self.up3(x4)
        y3 = self.dec3(torch.cat([y3, x3], dim=1))

        y2 = self.up2(y3)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))

        y1 = self.up1(y2)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        mask = torch.sigmoid(self.out_conv(y1))  # segmentation

        return mask
