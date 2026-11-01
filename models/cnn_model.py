"""
AlzheimerNet: Convolutional Neural Network with Channel and Spatial Attention
Designed for 4-stage Alzheimer's Disease Classification from MRI scans.
Stages:
  0: Mild Demented
  1: Moderate Demented
  2: Non-Demented
  3: Very Mild Demented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention Module to emphasize informative feature channels."""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduction = max(in_planes // ratio, 8)
        self.fc1 = nn.Conv2d(in_planes, reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module to focus on key anatomical brain regions."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module combining Channel & Spatial Attention."""
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ConvBlock(nn.Module):
    """Double Convolution block with BatchNorm, LeakyReLU, and Residual shortcut."""
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        # Shortcut for residual connection if channels change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAMBlock(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_attention:
            out = self.attention(out)
        out += residual
        return self.act(out)


class AlzheimerCNN(nn.Module):
    """
    Deep Convolutional Neural Network with Attention for Alzheimer's Stage Classification.
    Accepts 1-channel (grayscale) or 3-channel (RGB) MRI scans.
    """
    def __init__(self, in_channels=3, num_classes=4, dropout_rate=0.4):
        super(AlzheimerCNN, self).__init__()
        self.classes = [
            "Mild Impairment (Mild Demented)",
            "Moderate Impairment (Moderate Demented)",
            "Normal Control (Non-Demented)",
            "Early Impairment (Very Mild Demented)"
        ]
        
        # Stage 1: Feature Extraction (Input: [B, C, 224, 224])
        self.stage1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 112x112
        
        # Stage 2: Low-level structural patterns
        self.stage2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 56x56
        
        # Stage 3: Mid-level tissue and ventricle borders + Attention
        self.stage3 = ConvBlock(64, 128, use_attention=True)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 28x28
        
        # Stage 4: High-level atrophy markers + Attention
        self.stage4 = ConvBlock(128, 256, use_attention=True)
        self.pool4 = nn.MaxPool2d(2, 2)  # -> 14x14
        
        # Stage 5: Deep semantic representation + Attention
        self.stage5 = ConvBlock(256, 512, use_attention=True)
        # Target layer for Grad-CAM
        self.grad_cam_layer = self.stage5
        
        # Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(512, 128)
        self.fc_act = nn.LeakyReLU(0.1, inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.stage5(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_act(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def get_model(in_channels=3, num_classes=4):
    """Instantiate and return the AlzheimerNet model."""
    return AlzheimerCNN(in_channels=in_channels, num_classes=num_classes)


if __name__ == "__main__":
    model = AlzheimerCNN()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Model successfully built!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Batch size 2, 4 class logits)")
