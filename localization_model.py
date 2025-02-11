import os
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from augmentations import load_image, load_mask, apply_transforms

from torchvision.models import resnet34, ResNet34_Weights, convnext_base, ConvNeXt_Base_Weights
from pretrainedmodels import senet154
from torchvision.models.convnext import ConvNeXt, CNBlockConfig

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x_channel = x * ca
        # Spatial attention: concatenate max and mean across channel dimension
        sa = torch.cat([torch.max(x_channel, dim=1, keepdim=True)[0],
                        torch.mean(x_channel, dim=1, keepdim=True)], dim=1)
        sa = self.spatial_attention(sa)
        return x_channel * sa


class ConvNeXtBlockWithCBAM(nn.Module):
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True) if layer_scale > 0 else None
        self.cbam = CBAM(dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.cbam(x)  # Apply CBAM after depthwise conv
        # Permute to (N, H, W, C) for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # Permute back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return residual + x

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        # x is of shape [B, C, H, W]. We want to normalize across C.
        # Permute to [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        # Apply layer normalization (it expects last dim = normalized_shape)
        x = self.ln(x)
        # Permute back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + (skip_channels if skip_channels else 0), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)


    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        # First convolution block
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)

        # Regularization and final processing
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class LocalizationModel(nn.Module):
    
    # Supported encoder names: "resnet34", "senet154", "convnext_base"

    def __init__(self, encoder_name="convnext_base", pretrained=True, num_classes=1):
        super(LocalizationModel, self).__init__()
        self.encoder_name = encoder_name

        if encoder_name == "resnet34":
            encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.initial = nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                encoder.maxpool
            )
            self.layer1 = encoder.layer1
            self.layer2 = encoder.layer2
            self.layer3 = encoder.layer3
            self.layer4 = encoder.layer4
            encoder_channels = 512
            self.skip_channels = [256, 128, 64]  # Channels from layer3, layer2, layer1
        elif encoder_name == "senet154":
            encoder = senet154(pretrained=None)
            encoder.load_state_dict(torch.load("senet154-c7b49a05.pth"))
            self.encoder = encoder
            encoder_channels = 2048
            self.skip_channels = [1024, 512, 256]
        elif encoder_name == "convnext_base":
            self.encoder = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            depths=[3, 3, 9, 3]
            dims=[96, 192, 384, 768]
            in_channels=3
            # Stem: downsample input by factor 4.
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6)
            )
            # Stage 1
            self.stage1 = nn.Sequential(*[ConvNeXtBlockWithCBAM(dim=dims[0]) for _ in range(depths[0])])
            self.down1 = nn.Sequential(
                LayerNorm2d(dims[0], eps=1e-6),
                nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
            )
            # Stage 2
            self.stage2 = nn.Sequential(*[ConvNeXtBlockWithCBAM(dim=dims[1]) for _ in range(depths[1])])
            self.down2 = nn.Sequential(
                LayerNorm2d(dims[1], eps=1e-6),
                nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
            )
            # Stage 3
            self.stage3 = nn.Sequential(*[ConvNeXtBlockWithCBAM(dim=dims[2]) for _ in range(depths[2])])
            self.down3 = nn.Sequential(
                LayerNorm2d(dims[2], eps=1e-6),
                nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
            )
            # Stage 4
            self.stage4 = nn.Sequential(*[ConvNeXtBlockWithCBAM(dim=dims[3]) for _ in range(depths[3])])
            
            # For skip connections, we will use:
            # - skip1: output of stage1 (dims[0])
            # - skip2: output of stage2 (dims[1])
            # - skip3: output of stage3 (dims[2])
            # The encoder's final output is from stage4 (dims[3]).
            self.skip_channels = [dims[2], dims[1], dims[0]]
            encoder_channels = dims[3]
        else:
            raise ValueError("Unsupported encoder type: {}".format(encoder_name))

        # Decoder setup with proper skip connections
        self.decoder = nn.ModuleList([
            # First 3 blocks use skip connections
            DecoderBlock(encoder_channels, 256, skip_channels=self.skip_channels[0]),
            DecoderBlock(256, 128, skip_channels=self.skip_channels[1]),
            DecoderBlock(128, 64, skip_channels=self.skip_channels[2]),

            # Last 2 blocks without skip connections
            DecoderBlock(64, 32, skip_channels=0),
            DecoderBlock(32, 16, skip_channels=0),
        ])

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)
        self.final_activation = nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        if self.encoder_name == "resnet34":
            x0 = self.initial(x)  # 64x128x128
            x1 = self.layer1(x0)  # 64x128x128
            x2 = self.layer2(x1)  # 128x64x64
            x3 = self.layer3(x2)  # 256x32x32
            x4 = self.layer4(x3)  # 512x16x16
            skips = [x3, x2, x1]  # Skip connections for first 3 decoder blocks
        elif self.encoder_name == "senet154":
            x0 = self.encoder.layer0(x)  # Initial block (conv+bn+relu)
            x1 = self.encoder.layer1(x0)  # Layer1
            x2 = self.encoder.layer2(x1)  # Layer2
            x3 = self.encoder.layer3(x2)  # Layer3
            x4 = self.encoder.layer4(x3)  # Layer4
            skips = [x3, x2, x1]  # Skip connections for first 3 decoder blocks
        elif self.encoder_name == "convnext_base":
            x_stem = self.stem(x)               # [B, dims[0], H/4, W/4]
            x1 = self.stage1(x_stem)            # [B, dims[0], H/4, W/4]  -> skip3
            x_down1 = self.down1(x1)            # [B, dims[1], H/8, W/8]
            x2 = self.stage2(x_down1)           # [B, dims[1], H/8, W/8]  -> skip2
            x_down2 = self.down2(x2)            # [B, dims[2], H/16, W/16]
            x3 = self.stage3(x_down2)           # [B, dims[2], H/16, W/16] -> skip1
            x_down3 = self.down3(x3)            # [B, dims[3], H/32, W/32]
            x4 = self.stage4(x_down3)           # [B, dims[3], H/32, W/32]
            skips = [x3, x2, x1]  # Skip connections for first 3 decoder blocks
        else:
            x4 = self.encoder(x)
            skips = []

        # Decoder
        d = x4
        for i, decoder_block in enumerate(self.decoder):
            if i < len(skips):
                d = decoder_block(d, skips[i])
            else:
                d = decoder_block(d)

        out = self.final_conv(d)
        out = self.final_activation(out)
        return out


def combined_loss(pred, target, alpha=0.5, beta=0.7, gamma=2):
    # Dice Loss Component
    dice_loss = 1 - (2 * torch.sum(pred * target) + 1e-6) / \
                (torch.sum(pred + target) + 1e-6)

    # Focal Loss Component
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    focal_loss = torch.mean((1 - torch.exp(-bce)) ** gamma * bce)

    # Tversky Loss Component
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1 - target))
    fn = torch.sum((1 - pred) * target)
    tversky_loss = 1 - (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)

    # Balanced combination
    return 0.4 * dice_loss + 0.3 * focal_loss + 0.3 * tversky_loss


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    images_dir = os.path.join("geotiffs", "tier1", "images")
    labels_dir = os.path.join("geotiffs", "tier1", "labels")

    all_image_files = sorted([f for f in os.listdir(images_dir) if f.endswith("_pre_disaster.tif")])
    if not all_image_files:
        raise ValueError("No pre-disaster image files found in directory: {}".format(images_dir))

    # %20 of the dataset
    sample_size = max(1, len(all_image_files) // 5)
    sampled_files = all_image_files[:sample_size]
    print("Using {} out of {} images for this run.".format(sample_size, len(all_image_files)))

    model = LocalizationModel(encoder_name="resnet34", pretrained=True, num_classes=1)
    model.eval()

    losses = []
    for file_name in sampled_files:
        image_path = os.path.join(images_dir, file_name)
        label_file = file_name.replace("_pre_disaster.tif", "_pre_disaster.json")
        label_path = os.path.join(labels_dir, label_file)

        image = load_image(image_path)
        mask = load_mask(label_path, img_shape=image.shape[:2])
        transformed_image, transformed_mask = apply_transforms(image, mask)

        input_tensor = transformed_image.unsqueeze(0)
        output = model(input_tensor)

        loss_val = combined_loss(output, transformed_mask.unsqueeze(0))
        losses.append(loss_val.item())
        print(f"File: {file_name}, Loss: {loss_val.item():.4f}")

    if losses:
        avg_loss = sum(losses) / len(losses)
        print("Average Combined Loss on Sampled Data: {:.4f}".format(avg_loss))
