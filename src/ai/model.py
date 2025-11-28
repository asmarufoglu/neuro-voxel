import torch
import torch.nn as nn

class Simple3DUNet(nn.Module):
    
    def __init__(self, in_channels=4, out_channels=3):
        super(Simple3DUNet, self).__init__()

        #encoder
        self.encoder1 = self.conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        #bottleneck
        self.bottleneck = self.conv_block(64, 128)

        #decoder
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64, 32)

        #final
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        #encoder flow
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        #bottleneck
        b = self.bottleneck(p2)

        #decoder flow with skip connections
        d2 = self.upconv2(b)

        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.final_conv(d1)