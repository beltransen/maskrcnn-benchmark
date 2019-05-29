# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Variant of the resnet module that takes cfg as an argument.
Example usage. Strings may be specified in the config file.
    model = ResNet(
        "StemWithFixedBatchNorm",
        "BottleneckWithFixedBatchNorm",
        "ResNet50StagesTo4",
    )
OR:
    model = ResNet(
        "StemWithGN",
        "BottleneckWithGN",
        "ResNet50StagesTo4",
    )
Custom implementations may be written in user code and hooked in via the
`register_*` functions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn import Conv3d
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry


# ResNet stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_count",  # Number of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard ResNet models
# -----------------------------------------------------------------------------
# ResNet-50 (including all stages)
ResNet50StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, False), (4, 3, True))
)
# ResNet-50 up to stage 4 (excludes stage 5)
ResNet50StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 6, True))
)
# ResNet-101 (including all stages)
ResNet101StagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, False), (4, 3, True))
)
# ResNet-101 up to stage 4 (excludes stage 5)
ResNet101StagesTo4 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, False), (2, 4, False), (3, 23, True))
)
# ResNet-50-FPN (including all stages)
ResNet50FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 6, True), (4, 3, True))
)
# ResNet-101-FPN (including all stages)
ResNet101FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 4, True), (3, 23, True), (4, 3, True))
)
# ResNet-152-FPN (including all stages)
ResNet152FPNStagesTo5 = tuple(
    StageSpec(index=i, block_count=c, return_features=r)
    for (i, c, r) in ((1, 3, True), (2, 8, True), (3, 36, True), (4, 3, True))
)

class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        # self.cfg = cfg.clone()

        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.return_features = {}
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]
            module = _make_stage(
                transformation_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage_spec.block_count,
                num_groups,
                cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                first_stride=int(stage_spec.index > 1) + 1,
                dcn_config={
                    "stage_with_dcn": stage_with_dcn,
                    "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                    "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                },
                non_local=True
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


class ResNetHead(nn.Module):
    def __init__(
        self,
        block_module,
        stages,
        num_groups=1,
        width_per_group=64,
        stride_in_1x1=True,
        stride_init=None,
        res2_out_channels=256,
        dilation=1,
        dcn_config={}
    ):
        super(ResNetHead, self).__init__()

        stage2_relative_factor = 2 ** (stages[0].index - 1)
        stage2_bottleneck_channels = num_groups * width_per_group
        out_channels = res2_out_channels * stage2_relative_factor
        in_channels = out_channels // 2
        bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor

        block_module = _TRANSFORMATION_MODULES[block_module]

        self.stages = []
        stride = stride_init
        for stage in stages:
            name = "layer" + str(stage.index)
            if not stride:
                stride = int(stage.index > 1) + 1
            module = _make_stage(
                block_module,
                in_channels,
                bottleneck_channels,
                out_channels,
                stage.block_count,
                num_groups,
                stride_in_1x1,
                first_stride=stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
            stride = None
            self.add_module(name, module)
            self.stages.append(name)
        self.out_channels = out_channels

    def forward(self, x):
        for stage in self.stages:
            x = getattr(self, stage)(x)
        return x


def _make_stage(
    transformation_module,
    in_channels,
    bottleneck_channels,
    out_channels,
    block_count,
    num_groups,
    stride_in_1x1,
    first_stride,
    dilation=1,
    dcn_config={},
    non_local=False
):
    blocks = []
    stride = first_stride
    for idx in range(block_count):
        blocks.append(
            transformation_module(
                in_channels,
                bottleneck_channels,
                out_channels,
                num_groups,
                stride_in_1x1,
                stride,
                dilation=dilation,
                dcn_config=dcn_config
            )
        )
        stride = 1
        in_channels = out_channels

        if non_local and idx == (block_count-2): # Add non-local module
            blocks.append(NLB3D(out_channels, bottleneck_channels, 7, bn_layer=False))

    return nn.Sequential(*blocks)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups,
        stride_in_1x1,
        stride,
        dilation,
        norm_func,
        dcn_config
    ):
        super(Bottleneck, self).__init__()
        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(1, down_stride, down_stride), padding=0, bias=False),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, torch.nn.Conv3d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1  # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv3d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=(1, stride_1x1, stride_1x1),
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d( # TODO Upgrade to Conv3d
                bottleneck_channels, 
                bottleneck_channels, 
                with_modulated_dcn=with_modulated_dcn, 
                kernel_size=3,
                stride=stride_3x3,
                groups=num_groups,
                dilation=dilation,
                deformable_groups=deformable_groups,
                bias=False
            )
        else:
            self.conv2 = torch.nn.Conv3d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(1, 3, 3),
                stride=(1, stride_3x3, stride_3x3),
                padding=(0, dilation, dilation),
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = torch.nn.Conv3d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

        # print("USING Bottleneck")  # TODO Check which block is used in ResNet

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu_(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu_(out)
        print(out.shape)
        return out


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = torch.nn.Conv3d(
            3, out_channels, kernel_size=(1,7,7), stride=(1,2,2), padding=(0, 3, 3), bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

        print("USING BaseStem") # TODO Check which block is used in ResNet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        return x


class BottleneckWithFixedBatchNorm(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithFixedBatchNorm, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=torch.nn.BatchNorm3d,
            dcn_config=dcn_config
        )

        # print("USING BottleneckWithFixedBatchNorm") # TODO Check which block is used in ResNet


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class BottleneckWithGN(Bottleneck):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        out_channels,
        num_groups=1,
        stride_in_1x1=True,
        stride=1,
        dilation=1,
        dcn_config={}
    ):
        super(BottleneckWithGN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=group_norm,
            dcn_config=dcn_config
        )
        print("USING BottleneckWithGN") # TODO Check which block is used in ResNet


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)
        print("USING StemWithGN") # TODO Check which block is used in ResNet


class NLB3D(torch.nn.Module):
    def __init__(self, in_channels, h_channels, patch_size, bn_layer=True):
        super(NLB3D, self).__init__()
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.patch_size = patch_size
        self.theta = torch.nn.Conv3d(in_channels, h_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.psi = torch.nn.Conv3d(in_channels, h_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.g = torch.nn.Conv3d(in_channels, h_channels, kernel_size=1, stride=1, padding=0, bias=False)

        #     torch.nn.init.xavier_uniform(self.theta.weight)
        #     torch.nn.init.xavier_uniform(self.psi.weight)
        #     torch.nn.init.xavier_uniform(self.g.weight)
        torch.nn.init.constant_(self.theta.weight, 2)
        torch.nn.init.constant_(self.psi.weight, 0.4)
        torch.nn.init.constant_(self.g.weight, 0.01)

        if bn_layer:
            self.W = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=h_channels, out_channels=in_channels,
                                kernel_size=1, stride=1, padding=0),
                torch.nn.BatchNorm3d(in_channels)
            )
            torch.nn.init.constant_(self.W[1].weight, 0)
            torch.nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = torch.nn.Conv3d(in_channels=h_channels, out_channels=in_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)
            #       torch.nn.init.constant_(self.W.weight, 0)
            #       torch.nn.init.constant_(self.W.bias, 0)
            #       torch.nn.init.xavier_uniform(self.W.weight)
            torch.nn.init.constant_(self.W.weight, 0.01)

    def forward(self, x):
        print("NLB input: ", x.shape)
        assert (x.shape[2] % 2 != 0)

        #     out_theta = self.theta(x) # To get attention map for every stacked frames
        x_i = x[:, :, x.shape[2] // 2:x.shape[2] // 2 + 1, :, :]  # Current frame
        theta_x_i = self.theta(x_i)
        theta_x_i = theta_x_i.unsqueeze(-1)  # Add a dimension at the end
        theta_x_i = theta_x_i.view(theta_x_i.shape[0], theta_x_i.shape[1], -1, theta_x_i.shape[-1])
        #     print("theta_x_i: ", theta_x_i.shape)

        padding = self.patch_size // 2
        #     padding = 0 # TODO REMOVE
        #     x_pad = F.pad(x, (padding,padding,padding,padding,padding,padding)) # TODO Review T dimension padding (activated now)
        x_pad = F.pad(x, (padding, padding, padding, padding, 0, 0),
                      mode='replicate')  # TODO Review T dimension padding (deactivated now)
        #     patches  = x_pad.unfold(2, 3, 1)
        #     print('patches size:', patches.shape)
        patches = x_pad.unfold(3, self.patch_size, 1)
        #     print('patches size:', patches.shape)
        patches = patches.unfold(4, self.patch_size, 1)
        #     patches = x_pad.unfold(2, 3, 1).unfold(3, self.patch_size, 1).unfold(4, self.patch_size, 1)
        #     patches = patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2], -1, patches.shape[-2] * patches.shape[-1])
        #     print('patches size:', patches.shape)
        patches = patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2],
                                            patches.shape[3] * patches.shape[4], -1)
        #     print('patches size:', patches.shape)

        psi_patches = self.psi(patches)
        psi_patches = psi_patches.permute(0, 1, 3, 2, 4)
        psi_patches = psi_patches.contiguous().view(psi_patches.shape[0], psi_patches.shape[1], psi_patches.shape[2],
                                                    -1)
        g_patches = self.g(patches)
        g_patches = g_patches.permute(0, 1, 3, 2, 4)
        g_patches = g_patches.contiguous().view(g_patches.shape[0], g_patches.shape[1], g_patches.shape[2], -1)

        #     print("psi_patches: ", psi_patches.shape)
        #     print("g_patches: ", g_patches.shape)

        theta_x_i = theta_x_i.permute(0, 2, 3, 1)
        psi_patches = psi_patches.permute(0, 2, 1, 3)
        g_patches = g_patches.permute(0, 2, 3, 1)

        #     print("theta_x_i: ", theta_x_i.shape)
        #     print("psi_patches: ", psi_patches.shape)
        #     print("g_patches: ", g_patches.shape)

        att_map = torch.einsum('bpij,bpjk->bpik', theta_x_i, psi_patches)
        att_map2 = F.softmax(att_map, dim=3)
        #     print("att_map: ", att_map2.shape)
        #     print("att_map ", att_map2)
        y = torch.einsum('bpij,bpjk->bpik', att_map2, g_patches)
        y = y.squeeze(2)
        y = y.view(x_i.shape[0], self.h_channels,
                   x_i.shape[2], x_i.shape[3], x_i.shape[4])

        W_y = self.W(y)
        # print("y ", W_y.shape)
        z = x + W_y  # skip connection TODO Check summation (att map computed for middle channel and added to all of them)
        print('NLB output: ', z.shape)
        #     print('theta: ', self.theta.weight)
        return z, W_y, theta_x_i, psi_patches, g_patches



_TRANSFORMATION_MODULES = Registry({
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_STAGE_SPECS = Registry({
    "R-50-C4": ResNet50StagesTo4,
    "R-50-C5": ResNet50StagesTo5,
    "R-101-C4": ResNet101StagesTo4,
    "R-101-C5": ResNet101StagesTo5,
    "R-50-FPN": ResNet50FPNStagesTo5,
    "R-50-FPN-RETINANET": ResNet50FPNStagesTo5,
    "R-101-FPN": ResNet101FPNStagesTo5,
    "R-101-FPN-RETINANET": ResNet101FPNStagesTo5,
    "R-152-FPN": ResNet152FPNStagesTo5,
})
