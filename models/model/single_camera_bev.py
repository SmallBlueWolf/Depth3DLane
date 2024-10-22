import torch
import torchvision as tv
import torch.nn.functional as F
from torch import nn


def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


class ResNet34_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34_Backbone, self).__init__()
        resnet = tv.models.resnet34(pretrained=pretrained)

        # Initial layers
        self.conv1 = resnet.conv1  # (B, 64, H/2, W/2)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # (B, 64, H/4, W/4)

        # ResNet layers
        self.layer1 = resnet.layer1  # (B, 64, H/4, W/4)
        self.layer2 = resnet.layer2  # (B, 128, H/8, W/8)
        self.layer3 = resnet.layer3  # (B, 256, H/16, W/16)
        self.layer4 = resnet.layer4  # (B, 512, H/32, W/32)

        # Initialize weights
        for m in self.modules():
            naive_init_module(m)

    def forward(self, x):
        skips = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # skips[0]: After maxpool (B, 64, H/4, W/4)
        skips.append(x)

        x = self.layer1(x)
        # skips[1]: After layer1 (B, 64, H/4, W/4)
        skips.append(x)

        x = self.layer2(x)
        # skips[2]: After layer2 (B, 128, H/8, W/8)
        skips.append(x)

        x = self.layer3(x)
        # skips[3]: After layer3 (B, 256, H/16, W/16)
        skips.append(x)

        x = self.layer4(x)
        # skips[4]: After layer4 (B, 512, H/32, W/32)
        skips.append(x)

        return skips  # List of feature maps from different stages


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class DepthHeadUNet(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, base_channels=256):
        super(DepthHeadUNet, self).__init__()

        # Decoder layers
        self.up1 = self.up_block(in_channels, base_channels)  # From layer4 to layer3
        self.conv1 = self.conv_block(base_channels * 2, base_channels)

        self.up2 = self.up_block(base_channels, base_channels // 2)  # From layer3 to layer2
        self.conv2 = self.conv_block(base_channels, base_channels // 2)

        self.up3 = self.up_block(base_channels // 2, base_channels // 4)  # From layer2 to layer1
        self.conv3 = self.conv_block(base_channels // 2, base_channels // 4)

        self.up4 = self.up_block(base_channels // 4, base_channels // 8)  # From layer1 to after maxpool
        self.conv4 = self.conv_block(base_channels // 8 + 64, base_channels // 8)  # +64 channels from after maxpool

        # Final upsampling to reach (576, 1024)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)  # From (B, 32, 288, 512) to (B, 32, 576, 1024)
        self.final_conv = nn.Conv2d(base_channels // 8, out_channels, kernel_size=1)
        naive_init_module(self.final_conv)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skips):
        """
        x: Input feature map from backbone (B, 512, H/32, W/32)
        skips: List of feature maps from backbone for skip connections
               [after maxpool, layer1, layer2, layer3, layer4]
        """
        # Decoder step 1: Upsample layer4 to layer3's spatial size
        d1 = self.up1(x)  # (B, 256, H/16, W/16)
        s1 = skips[3]  # Corresponding skip from layer3 (B, 256, H/16, W/16)
        d1 = torch.cat([d1, s1], dim=1)  # (B, 512, H/16, W/16)
        d1 = self.conv1(d1)  # (B, 256, H/16, W/16)

        # Decoder step 2: Upsample to layer2's spatial size
        d2 = self.up2(d1)  # (B, 128, H/8, W/8)
        s2 = skips[2]  # Corresponding skip from layer2 (B, 128, H/8, W/8)
        d2 = torch.cat([d2, s2], dim=1)  # (B, 256, H/8, W/8)
        d2 = self.conv2(d2)  # (B, 128, H/8, W/8)

        # Decoder step 3: Upsample to layer1's spatial size
        d3 = self.up3(d2)  # (B, 64, H/4, W/4)
        s3 = skips[1]  # Corresponding skip from layer1 (B, 64, H/4, W/4)
        d3 = torch.cat([d3, s3], dim=1)  # (B, 128, H/4, W/4)
        d3 = self.conv3(d3)  # (B, 64, H/4, W/4)

        # Decoder step 4: Upsample to after maxpool's spatial size
        d4 = self.up4(d3)  # (B, 32, H/2, W/2)
        s4 = skips[0]  # Corresponding skip from after maxpool (B, 64, H/4, W/4)
        # To match spatial dimensions, upsample s4
        s4 = F.interpolate(s4, size=d4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, s4], dim=1)  # (B, 96, H/2, W/2)
        d4 = self.conv4(d4)  # (B, 32, H/2, W/2)

        # Final upsampling
        out = self.final_up(d4)  # (B, 32, H, W)
        out = self.final_conv(out)  # (B, 1, H, W)
        depth_map = torch.sigmoid(out)  # Normalize to [0, 1]

        return depth_map


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])  # 这个 B,V,C,H*W
        bev_view = self.fc_transform(x)  # 拿出一个视角
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# model
# ResNet34 骨干网络 (self.bb)，在 ImageNet 上进行预训练。
# 一个下采样层 (self.down)，用于减小特征图的空间维度。
# 两个全连接变换层 (self.s32transformer 和 self.s64transformer)，将 ResNet 骨干网络的特征图转换为 BEV 表示。
# 车道线检测头 (self.lane_head)，以 BEV 表示作为输入，输出表示检测到的车道线的张量。
# 可选的 2D 图像车道线检测头 (self.lane_head_2d)，以 ResNet 骨干网络的输出作为输入，输出表示原始图像中检测到的车道线的张量。
class BEV_LaneDet(nn.Module):
    def __init__(self, bev_shape, output_2d_shape, train=True, depth_label=True):
        """

        :param bev_shape:
        :param output_2d_shape:
        :param train:
        :param depth_label: 是否启用 depth auxiliary head
        """
        super(BEV_LaneDet, self).__init__()

        # Custom ResNet34 Backbone with Skip Connections
        self.backbone = ResNet34_Backbone(pretrained=True)

        # Spatial Transformers to BEV
        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)
                ),
                downsample=nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            )
        )

        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 5))
        self.s64transformer = FCTransform_((1024, 9, 16), (256, 25, 5))
        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.depth_label = depth_label
        # Depth Head with Skip Connections
        if self.depth_label:
            self.depth_head = DepthHeadUNet(in_channels=512, out_channels=1, base_channels=256)

        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512)

    def forward(self, img):
        # Backbone feature extraction with skip connections
        skips = self.backbone(img)  # List of feature maps [After maxpool, layer1, layer2, layer3, layer4]

        img_s32 = skips[-1]  # layer4 output: (B, 512, 18, 32)
        img_s64 = self.down(img_s32)  # (B, 1024, 9, 16)

        # Spatial transformers to BEV
        bev_32 = self.s32transformer(img_s32)  # (B, 256, 25, 5)
        bev_64 = self.s64transformer(img_s64)  # (B, 256, 25, 5)
        bev = torch.cat([bev_64, bev_32], dim=1)  # (B, 512, 25, 5)

        # Lane head predictions
        lane_outputs = self.lane_head(bev)  # (ms_new, me_new, m_offset_new, m_z)

        if self.depth_label:
            # Depth prediction with skip connections
            depth_map = self.depth_head(img_s32, skips)  # (B, 1, 576, 1024)
        else:
            depth_map = None

        if self.is_train:
            lane_2d_outputs = self.lane_head_2d(img_s32)  # (ms, me)
            return lane_outputs, lane_2d_outputs, depth_map
        else:
            return lane_outputs

