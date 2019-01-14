import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

'''
Audio analysis network
U-Net (https://arxiv.org/pdf/1505.04597.pdf) with modifications
Input: torch.Tensor of shape [1,1,256,256]
Output: torch.Tensor of shape [1,16,256,256]
'''
K = 16
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )

def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2
    )

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1
    )

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv = double_conv(self.in_channels, self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = self.double_conv(input)
        x = self.pool(x), x
        return x # height and width cut in half,
                 # num of channels changes from in_channels to out_channels

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv = up_conv(self.in_channels, self.out_channels)
        self.double_conv = double_conv(self.in_channels, self.out_channels)

    def forward(self, encoder_out, decoder_out):
        decoder_out = self.deconv(decoder_out)
        x = torch.cat((encoder_out, decoder_out), 1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, start_channels=16, out_channels=K, depth=6):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.start_channels = start_channels
        self.out_channels = out_channels
        self.depth = depth

        self.downs = deque()
        for i in range(self.depth):
            if i == 0:
                self.downs.append(Down(self.in_channels, self.start_channels))
            else:
                self.downs.append(Down(self.start_channels*(2**(i-1)), self.start_channels*(2**i)))
        # After down convolutions, the shape of x is torch.Size([1,512,4,4])
        self.double_conv = double_conv(self.start_channels*(2**(self.depth-1)), self.start_channels*(2**self.depth))
        # After double_conv, the shape of x is torch.Size([1,1024,4,4])
        self.ups = deque()
        for i in range(self.depth):
            self.ups.appendleft(Up(self.start_channels*(2**(i+1)), self.start_channels*(2**i)))
        self.final_conv = nn.Conv2d(self.start_channels, self.out_channels, 1)

    def forward(self, x):
        encoder_outs = deque()
        for _, model in enumerate(self.downs):
            x, encoder_out = model(x)
            encoder_outs.appendleft(encoder_out)
        x = self.double_conv(x)
        for idx, model in enumerate(self.ups):
            encoder_out = encoder_outs[idx]
            x = model(encoder_out, x)
        x = self.final_conv(x)
        x = x.squeeze(0)
        return x

'''
Video analysis network
ResNet-18 (https://arxiv.org/abs/1512.03385) with modifications
Input: torch.Tensor of shape [T,3,H,W]
Output: torch.Tensor of shape [T,K,H/16,W/16] -> [K,H,W] visual feature of size K for each pixel
'''
def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, padding=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, padding, dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels) # ??
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # Remove the stride of the last residual block, and make
        # the convolution layers in this block to have a dilation of 2.
        # Add padding to keep the shape of input and output same.
        self.layer4 = self._make_layer(block, 512, layers[3], padding=2, dilation=2)
        self.last_conv = conv3x3(512, K)
        self.transpose_conv = nn.ConvTranspose2d(K, K, 16, stride=16)

    def _make_layer(self, block, out_channels, blocks, stride=1, padding=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, padding, dilation, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x is a torch.Tensor of shape [T,3,H,W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.last_conv(x) # torch.Tensor of shape [T,K,H/16,W/16]

        x = x.max(0)[0]   # Temporal max pooling
        x = x.unsqueeze(0) # torch.Tensor of shape [1,K,H/16,W/16]
        x = self.transpose_conv(x)  # Spatial unpooling
        x = x.squeeze(0) # torch.Tensor of shape [K,H,W]
        return x

'''
Audio synthesizer network.
A linear layer.
Input: audio feature (torch.Tensor of shape [K,256,256]) and visual feature (torch.Tensor of shape [K,256,256])
Output: spectrogram mask (torch.Tensor of shape [256,256])
'''
class Synthesizer(nn.Module):
    def __init__(self, bias=True):
        super(Synthesizer, self).__init__()
        self.bias = bias
        self.linear = nn.Linear(K, 1, bias=self.bias)

    def forward(self, x, y):
        out = torch.mul(x,y)
        out = out.transpose(0,1).transpose(1,2)
        out = self.linear(out)
        out = out.squeeze(2)
        return out

class LocalizationNet(nn.Module):
    def __init__(self, in_channels=1, start_channels=16, out_channels=K, depth=6, block=BasicBlock, layers=[2,2,2,2], bias=True):
        super(LocalizationNet, self).__init__()
        self.in_channels = in_channels
        self.start_channels = start_channels
        self.out_channels = out_channels
        self.depth = depth
        self.block = block
        self.layers = layers
        self.bias = bias
        self.audio_net = UNet(in_channels=self.in_channels, start_channels=self.start_channels, out_channels=self.out_channels, depth=self.depth)
        self.visual_net = ResNet(self.block, self.layers)
        self.synthesizer = Synthesizer(self.bias)

    def forward(self, audio_feature, visual_feature):
        x = self.audio_net(audio_feature)
        y = self.visual_net(visual_feature)
        out = self.synthesizer(x,y)
        return out
