import torch
import torch.nn as nn
import params

class res_block(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        
        self.relu1 = nn.ReLU() # bn_relu(c_in)
        self.conv1 = nn.Conv3d(c_in, c_out, kernel_size=params.nn_kernel, padding=1, stride=stride )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv3d(c_out, c_out, kernel_size=params.nn_kernel, padding=1, stride=1)
        self.conv3 = nn.Conv3d(c_in, c_out, kernel_size=1, padding=0, stride=stride)
        
    def forward(self, inp):
        
        x = self.relu1(inp)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x2 = self.conv3(inp)
        
        return x+x2


class dec_block(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu1 = nn.ReLU()
        self.res_block = res_block(c_in + c_out, c_out)
        
    def forward(self, dec_inp, enc_out):
        x = self.upsample(dec_inp)
        x = torch.cat([x, enc_out], axis=1)
        x = self.res_block(x)
        return x




class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv3d(params.tk*2, 64, kernel_size=params.nn_kernel, padding=1)  # bn_relu(c_in)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(64, 64, kernel_size=params.nn_kernel, padding=1)
        self.conv3 = nn.Conv3d(params.tk*2, 64, kernel_size=1, padding=0) # 1x1 conv
        
        self.enc1 = res_block(64, 128, stride=2)
        self.enc2 = res_block(128, 256, stride=2)
        
        self.enc3 = res_block(256, 512, stride=2)
        
        self.dec1 = dec_block(512, 256)
        self.dec2 = dec_block(256, 128)
        self.dec3 = dec_block(128, 64)
        
        self.conv4 = nn.Conv3d(64, params.tk*2, kernel_size=1, padding=0)
    def forward(self, inp):
        
        """
        Enc
        """
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.conv2(x)
        x2 = self.conv3(inp)
        enc1 = x + x2
        
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)
        
        """
        Bridge
        """
        bridge = self.enc3(enc3)
        
        """
        Dec
        """
        dec1 = self.dec1(bridge, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        
        out = self.conv4(dec3)
        return out