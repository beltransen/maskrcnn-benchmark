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
from collections import namedtuple, OrderedDict

import torch
import torch.nn.functional as F
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
        self.cfg = cfg.clone()

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
                }
            )
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features

            # Add dummy identity layer TODO REMOVE
            # identity_name = "identity" + str(stage_spec.index)
            # self.add_module(identity_name, Identity2D())
            # self.stages.append(identity_name)
            # self.return_features[identity_name] = False


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
        # torch.manual_seed(0)
        # x = torch.rand(x.shape).cuda()
        # print('x', x)
        # print('x', x.shape)
        # print('FORWARDIIIING')
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            # print(getattr(self, stage_name))
            # print('{}: {}'.format(stage_name, x.shape)) # TODO REMOVE
            if self.return_features[stage_name]:
                outputs.append(x)

        # print('out', outputs)
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
    for _ in range(block_count):
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
                Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=False
                ),
                norm_func(out_channels),
            )
            for modules in [self.downsample,]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)

        if dilation > 1:
            stride = 1 # reset to be 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
        )
        self.bn1 = norm_func(bottleneck_channels)
        # TODO: specify init for the above
        with_dcn = dcn_config.get("stage_with_dcn", False)
        if with_dcn:
            deformable_groups = dcn_config.get("deformable_groups", 1)
            with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
            self.conv2 = DFConv2d(
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
            self.conv2 = Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                stride=stride_3x3,
                padding=dilation,
                bias=False,
                groups=num_groups,
                dilation=dilation
            )
            nn.init.kaiming_uniform_(self.conv2.weight, a=1)

        self.bn2 = norm_func(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = norm_func(out_channels)

        for l in [self.conv1, self.conv3,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

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

        return out


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

        for l in [self.conv1,]:
            nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
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
            norm_func=FrozenBatchNorm2d,
            dcn_config=dcn_config
        )

class BottleneckBN(Bottleneck):
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
        super(BottleneckBN, self).__init__(
            in_channels=in_channels,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            stride_in_1x1=stride_in_1x1,
            stride=stride,
            dilation=dilation,
            norm_func=torch.nn.BatchNorm2d,
            dcn_config=dcn_config
        )

class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class StemBN(BaseStem):
    def __init__(self, cfg):
        super(StemBN, self).__init__(
            cfg, norm_func=torch.nn.BatchNorm2d
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


class StemWithGN(BaseStem):
    def __init__(self, cfg):
        super(StemWithGN, self).__init__(cfg, norm_func=group_norm)


class Identity2D(nn.Module): # TODO REMOVE
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """
    def __init__(self, *args, **kwargs):
        super(Identity2D, self).__init__()

    def forward(self, input):
        return input


class Identity(nn.Module): # TODO REMOVE if Pytorch 1.1 used
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input[0][:,:,0,:,:]


class ResNetNL(nn.Module):
    def __init__(self, cfg):
        super(ResNetNL, self).__init__()

        # If we want to use the cfg in forward(), then we should make a copy
        # of it and store it for later use:
        self.cfg = cfg.clone()
        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        transformation_module = _TRANSFORMATION_MODULES[cfg.MODEL.RESNETS.TRANS_FUNC]
        nonlocal_ctx_module = _NON_LOCAL_CTX_MODULES[cfg.NON_LOCAL_CTX.MODULE]

        self.ctx_first = cfg.NON_LOCAL_CTX.FIRST_CTX_FRAME
        self.return_attention_maps = cfg.NON_LOCAL_CTX.RETURN_ATTENTION

        # Construct the stem module
        self.stem = stem_module(cfg)

        # Constuct the specified ResNet stages
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
        stage2_bottleneck_channels = num_groups * width_per_group
        stage2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        self.stages = []
        self.blocks1 = nn.ModuleList()
        self.nonlocals = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.return_features = {}
        self.nl_block = {}

        cuda_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count()) ]
        current_device = 0

        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            stage2_relative_factor = 2 ** (stage_spec.index - 1)
            bottleneck_channels = stage2_bottleneck_channels * stage2_relative_factor
            out_channels = stage2_out_channels * stage2_relative_factor
            stage_with_dcn = cfg.MODEL.RESNETS.STAGE_WITH_DCN[stage_spec.index -1]
            non_local_enabled = cfg.NON_LOCAL_CTX.ENABLED if stage_spec.index in cfg.NON_LOCAL_CTX.STAGES else False

            # Part 1
            block1 = []
            stride = int(stage_spec.index > 1) + 1
            for _ in range(stage_spec.block_count-1):
                block1.append(
                    transformation_module(
                        in_channels,
                        bottleneck_channels,
                        out_channels,
                        num_groups,
                        cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                        stride,
                        dilation=1,
                        dcn_config={
                            "stage_with_dcn": stage_with_dcn,
                            "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                            "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                        }
                    )
                )
                stride = 1
                in_channels = out_channels

            self.blocks1.add_module(name, nn.Sequential(*block1).to(cuda_devices[current_device]))

            # NL
            layer_conf = cfg.NON_LOCAL_CTX
            if non_local_enabled and cfg.NON_LOCAL_CTX.POSITION is "after1x1":
                self.nonlocals.add_module(name, nonlocal_ctx_module(
                    out_channels, cfg.NON_LOCAL_CTX.BOTTLENECK_RATIO, cfg.NON_LOCAL_CTX.ZEROS_INIT,
                    return_attention=self.return_attention_maps, extra_args=layer_conf).to(cuda_devices[current_device]))
            else:
                self.nonlocals.add_module(name, Identity().to(cuda_devices[current_device]))

            # Part 2
            block2 = transformation_module(
                        out_channels,
                        bottleneck_channels,
                        out_channels,
                        num_groups,
                        cfg.MODEL.RESNETS.STRIDE_IN_1X1,
                        stride,
                        dilation=1,
                        dcn_config={
                            "stage_with_dcn": stage_with_dcn,
                            "with_modulated_dcn": cfg.MODEL.RESNETS.WITH_MODULATED_DCN,
                            "deformable_groups": cfg.MODEL.RESNETS.DEFORMABLE_GROUPS,
                        }
                    )
            self.blocks2.add_module(name, nn.Sequential(OrderedDict([(str(stage_spec.block_count-1), block2),]))
                                    .to(cuda_devices[current_device]))

            if non_local_enabled and cfg.NON_LOCAL_CTX.POSITION is "afterAdd":
                exit('[ERROR] AfterAdd position for NL context module not yet implemented')

            # self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
            self.nl_block[name] = non_local_enabled

            # Update device number
            current_device = (current_device+1) % torch.cuda.device_count()

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
                for p in m.parameters():
                    p.requires_grad = False
            else:
                # print('FREEZING')
                name = "layer" + str(stage_index)
                m1 = getattr(self.blocks1, name)
                # print(m1)
                for p in m1.parameters():
                    p.requires_grad = False
                # if self.nl_block[name]:
                m2 = getattr(self.nonlocals, name)
                # print(m2)
                for p in m2.parameters():
                    p.requires_grad = False
                m3 = getattr(self.blocks2, name)
                # print(m3)
                for p in m3.parameters():
                    p.requires_grad = False

    def forward(self, x):
        outputs = []
        #
        # print('INPUT SHAPE: ', x.shape)
        # print('INPUT DIMS: ', len(x.shape))
        if len(x.shape) == 4:
            x = x.unsqueeze(2)

        # torch.manual_seed(0)
        # x[:, :, 0, :, :] = torch.rand(x[:, :, 0, :, :].shape)
        # print('x', x[:, :, 0, :, :])
        # print('x', x[:, :, 0, :, :].shape)
        # print('x', x.shape)
        xs = []
        att_maps = dict()
        for i in range(x.shape[2]):
            xs.append(self.stem(x[:, :, i, :, :]))

        for s in range(len(self.stages)):
            stage_name = self.stages[s]
            num_frames = x.shape[2]  # TODO DANGER!
            for i in range(num_frames):
                xs[i] = getattr(self.blocks1, stage_name)(xs[i])
                # print('[Blocks1] Forward (t-{}) {}: {}'.format(i, stage_name, getattr(self.blocks1, stage_name)))
                # print('[Blocks1] Forward (t-{}) {}: {}'.format(i, stage_name, xs[i].shape))
                # print(getattr(self.blocks1, stage_name))

            prev_x0 = xs[0]
            if self.nl_block[stage_name]:  # TODO Remove: Not necessary? Either Identity or NL module
                stacked = torch.stack(xs, 2)
                # print('stacked size: ', stacked.shape)
                print('[NL] Forward (t-{}) {}: {}'.format(i, stage_name, getattr(self.nonlocals, stage_name)))
                if self.return_attention_maps:  # TODO Add the capability to return attmaps to all nonlocal context layers
                    # print('returniing attmaps')
                    aux = getattr(self.nonlocals, stage_name)(stacked)
                    # aux = getattr(self.nonlocals, stage_name)(stacked[:, :, 0:1, :, :], stacked[:, :, self.ctx_first:, :, :])
                    if not isinstance(getattr(self.nonlocals, stage_name), Identity):
                        xs[0], att_map = aux[0], aux[1]
                        num_ctx_frames = x.shape[2] - self.ctx_first
                        att_map = att_map.view(att_map.shape[0], xs[0].shape[2], xs[0].shape[3],
                                               num_ctx_frames, att_map.shape[3]//num_ctx_frames)
                        att_maps[stage_name] = att_map
                else:
                     xs[0] = getattr(self.nonlocals, stage_name)(stacked)
                    # xs[0] = getattr(self.nonlocals, stage_name)(stacked[:, :, 0:1, :, :], stacked[:, :, self.ctx_first:, :, :])
                # print(getattr(self.nonlocals, stage_name))
                # if isinstance(getattr(self.nonlocals, stage_name), Identity): # TODO Review, it's not always equal!!
                #     print("identittyyy")
                #     print('prev', prev_x0.shape)
                #     print('xs', xs[0].shape)
                #     assert (torch.eq(prev_x0, xs[0]).all())

            # for i in range(x.shape[0]): # TODO Changed
            for i in range(num_frames):
                xs[i] = getattr(self.blocks2, stage_name)(xs[i])
                # print(getattr(self.blocks2, stage_name))
                # print('[Blocks2] Forward (t-{}) {}: {}'.format(i, stage_name, getattr(self.blocks2, stage_name)))
                # print('[Blocks2] Forward (t-{}) {}: {}'.format(i, stage_name, xs[i].shape))

            if self.return_features[stage_name]:
                outputs.append(xs[0])  # TODO Check: returns only the output of current frame [0] for FPN

        # print('Resnet outputs: ', len(outputs))
        # for i in range(len(outputs)):
        #     print('Shape output {}: {}'.format(i, outputs[i].shape))
        print('out ', outputs[3])

        if self.return_attention_maps:
            return outputs, att_maps
        else:
            return outputs


class RegionNonLocal3D(torch.nn.Module):
    """ Region Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        zeros_init (bool): Zero weight initialization for conv_out
        return_attention (bool): If True, attention map is returned
        extra_args (dict): For specific layer settings:
            - patch_size (int): Size of the region to be used
            - bn_layer (bool): If True, BatchNorm is used after conv_out
            - mode (str): Options are `embedded_gaussian` and `dot_product`.
    """
    def __init__(self, in_channels, reduction=2, zeros_init=True, return_attention=False, extra_args=None):
        super(RegionNonLocal3D, self).__init__()
        assert isinstance(extra_args, dict)
        self.in_channels = in_channels
        self.h_channels = in_channels // reduction
        self.zeros_init = zeros_init
        self.return_attention_map = return_attention
        self.patch_size = extra_args['PATCH_SIZE']
        self.bn_layer = extra_args['USE_BN']
        self.ctx_first = extra_args['FIRST_CTX_FRAME']
        self.mode = extra_args['MODE']
        assert self.mode in ['embedded_gaussian', 'dot_product']

        self.theta = torch.nn.Conv3d(in_channels, self.h_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.phi = torch.nn.Conv3d(in_channels, self.h_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.g = torch.nn.Conv3d(in_channels, self.h_channels, kernel_size=1, stride=1, padding=0, bias=False)

        if self.bn_layer:
            self.conv_out = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=self.h_channels, out_channels=in_channels,
                                kernel_size=1, stride=1, padding=0),
                torch.nn.BatchNorm3d(in_channels, momentum=0.9)
            )
        else:
            self.conv_out = torch.nn.Conv3d(in_channels=self.h_channels, out_channels=in_channels,
                                     kernel_size=1, stride=1, padding=0, bias=False)

        self.init_weights()

    def init_weights(self, std=0.01):
        for m in [self.g, self.theta, self.phi]:
            torch.nn.init.normal_(m.weight, std=std)
        if self.zeros_init:
            if self.bn_layer:
                torch.nn.init.constant_(self.conv_out[0].weight, 0)
            else:
                torch.nn.init.constant_(self.conv_out.weight, 0)
        else:
            if self.bn_layer:
                torch.nn.init.normal_init(self.conv_out[0].weight, std=std)
            else:
                torch.nn.init.normal_init(self.conv_out.weight, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.h_channels`
            pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        x_i, ctx = x[:, :, 0:1, :, :], x[:, :, self.ctx_first:, :, :]
        assert (x_i.shape[2] % 2 != 0)
        # print("NLB input: ", x_i.shape)
        # print("NLB ctx: ", ctx.shape)
        #     out_theta = self.theta(x) # To get attention map for every stacked frames
        # x_i = x[:, :, x.shape[2] // 2:x.shape[2] // 2 + 1, :, :]  # Current frame
        # x_i = x  # Current frame
        # assert (x.shape[2] % 2 != 0)
        # print("NLB input: ", x.shape)
        # print("NLB ctx: ", ctx.shape)
        theta_x_i = self.theta(x_i)
        theta_x_i = theta_x_i.unsqueeze(-1)  # Add a dimension at the end
        theta_x_i = theta_x_i.view(theta_x_i.shape[0], theta_x_i.shape[1], -1, theta_x_i.shape[-1])
        #     print("theta_x_i: ", theta_x_i.shape)

        padding = self.patch_size // 2

        ## TODO NEW APPROACH

        theta_x_i = theta_x_i.permute(0, 2, 3, 1)

        phi_x_i = self.phi(ctx)
        #         phi_x_i = F.max_pool3d(phi_x_i, kernel_size=(1,2,2), stride=(1,2,2))
        #         phi_x_i = phi_x_i.unsqueeze(-1)  # Add a dimension at the end
        #         phi_x_i = phi_x_i.view(phi_x_i.shape[0], phi_x_i.shape[1], -1, phi_x_i.shape[-1])
        phi_x_i_pad = F.pad(phi_x_i, (padding, padding, padding, padding, 0, 0),
                            mode='replicate')  # TODO Review T dimension padding (deactivated now)
        phi_patches = phi_x_i_pad.unfold(3, self.patch_size, 1)
        phi_patches = phi_patches.unfold(4, self.patch_size, 1)
        phi_patches = phi_patches.contiguous().view(phi_patches.shape[0], phi_patches.shape[1], phi_patches.shape[2],
                                                    phi_patches.shape[3] * phi_patches.shape[4], -1)

        phi_patches = phi_patches.permute(0, 3, 1, 2, 4)
        phi_patches = phi_patches.contiguous().view(phi_patches.shape[0], phi_patches.shape[1], phi_patches.shape[2],
                                                    -1)
        #         print('phi_patches size:', phi_patches.element_size() * phi_patches.nelement())

        g_x_i = self.g(ctx)
        # print('G size:', g_x_i.shape)
        #         g_x_i = F.max_pool3d(g_x_i, kernel_size=(1,2,2), stride=(1,2,2))
        # print('G pooled size:', g_x_i.shape)
        #         g_x_i = g_x_i.unsqueeze(-1)  # Add a dimension at the end
        #         g_x_i = g_x_i.view(g_x_i.shape[0], g_x_i.shape[1], -1, g_x_i.shape[-1])
        g_x_i_pad = F.pad(g_x_i, (padding, padding, padding, padding, 0, 0),
                          mode='replicate')  # TODO Review T dimension padding (deactivated now)
        g_patches = g_x_i_pad.unfold(3, self.patch_size, 1)
        g_patches = g_patches.unfold(4, self.patch_size, 1)
        g_patches = g_patches.contiguous().view(g_patches.shape[0], g_patches.shape[1], g_patches.shape[2],
                                                g_patches.shape[3] * g_patches.shape[4], -1)
        g_patches = g_patches.permute(0, 3, 1, 2, 4)
        g_patches = g_patches.contiguous().view(g_patches.shape[0], g_patches.shape[1], -1, g_patches.shape[2])
        #         print('g_patches size:', g_patches.element_size() * g_patches.nelement())

        # print('theta_x_i size:', theta_x_i.shape)
        # print('phi_x_i size:', phi_x_i.shape)
        # print('g_x_i size:', g_x_i.shape)
        # print('phi_patches size:', phi_patches.shape)
        # print('g_patches size:', g_patches.shape)

        ## TODO END NEW APPROACH


        ## TODO OLD APPROACH
        # #     padding = 0 # TODO REMOVE
        # #     x_pad = F.pad(x, (padding,padding,padding,padding,padding,padding)) # TODO Review T dimension padding (activated now)
        #
        # # CHANGED TO ctx instead of original 'x' tensor
        # x_pad = F.pad(ctx, (padding, padding, padding, padding, 0, 0), mode='replicate')  # TODO Review T dimension padding (deactivated now)
        # #     patches  = x_pad.unfold(2, 3, 1)
        # #     print('patches size:', patches.shape)
        # patches = x_pad.unfold(3, self.patch_size, 1)
        # #     print('patches size:', patches.shape)
        # patches = patches.unfold(4, self.patch_size, 1)
        # #     patches = x_pad.unfold(2, 3, 1).unfold(3, self.patch_size, 1).unfold(4, self.patch_size, 1)
        # #     patches = patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2], -1, patches.shape[-2] * patches.shape[-1])
        # #     print('patches size:', patches.shape)
        # patches = patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2],
        #                                     patches.shape[3] * patches.shape[4], -1)
        # #     print('patches size:', patches.shape)
        #
        # phi_patches = self.phi(patches)
        # phi_patches = phi_patches.permute(0, 1, 3, 2, 4)
        # phi_patches = phi_patches.contiguous().view(phi_patches.shape[0], phi_patches.shape[1], phi_patches.shape[2],
        #                                             -1)
        # g_patches = self.g(patches)
        # g_patches = g_patches.permute(0, 1, 3, 2, 4)
        # g_patches = g_patches.contiguous().view(g_patches.shape[0], g_patches.shape[1], g_patches.shape[2], -1)
        #
        # #     print("phi_patches: ", phi_patches.shape)
        # #     print("g_patches: ", g_patches.shape)
        #
        # theta_x_i = theta_x_i.permute(0, 2, 3, 1)
        # phi_patches = phi_patches.permute(0, 2, 1, 3)
        # g_patches = g_patches.permute(0, 2, 3, 1)

        ## TODO END OLD APPROACH

        #     print("theta_x_i: ", theta_x_i.shape)
        #     print("phi_patches: ", phi_patches.shape)
        #     print("g_patches: ", g_patches.shape)

        # # TODO REVIEW IF THIS SNIPPET CAN BE USED
        # pairwise_func = getattr(self, self.mode)
        # # pairwise_weight: [N, TxHxW, TxHxW]
        # pairwise_weight = pairwise_func(theta_x_i, phi_patches)
        #
        # # y: [N, TxHxW, C]
        # y = torch.matmul(pairwise_weight, g_patches)
        #
        # # END OF SNIPPET

        att_map = torch.einsum('bpij,bpjk->bpik', theta_x_i, phi_patches)
        att_map /= theta_x_i.shape[-1]**-0.5
        att_map2 = F.softmax(att_map, dim=-1)
        #     print("att_map: ", att_map2.shape)
        #     print("att_map ", att_map2)
        y = torch.einsum('bpij,bpjk->bpik', att_map2, g_patches)
        y = y.squeeze(2)
        y = y.view(x_i.shape[0], self.h_channels,
                   x_i.shape[2], x_i.shape[3], x_i.shape[4])

        W_y = self.conv_out(y)
        # print("y ", W_y.shape)
        # z = x + W_y  # skip connection TODO Check summation (att map computed for middle channel and added to all of them)
        z = x_i + W_y  # skip connection TODO Changed to x_i
        # print('NLB output: ', z.shape)
        #     print('theta: ', self.theta.weight)

        if self.return_attention_map:
            return z.squeeze(2), att_map2  #, W_y, theta_x_i, phi_patches, g_patches
        else:
            return z.squeeze(2)


class NonLocal3D(torch.nn.Module):
    """Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        zeros_init (bool): Zero weight initialization for conv_out
        return_attention (bool): If True, attention map is returned
        extra_args (dict): For specific layer settings:
            - use_scale (bool): Whether to scale pairwise_weight by 1/h_channels.
            - mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self, in_channels, reduction=2, zeros_init=True, return_attention=False, extra_args=None):
        super(NonLocal3D, self).__init__()
        self.in_channels = in_channels
        self.h_channels = in_channels // reduction
        self.zeros_init = zeros_init
        self.return_attention_map = return_attention
        self.use_scale = extra_args['USE_SCALE']
        self.mode = extra_args['MODE']
        assert self.mode in ['embedded_gaussian', 'dot_product']
        print('Mode: ', self.mode)

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = torch.nn.Conv3d(self.in_channels, self.h_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.theta = torch.nn.Conv3d(self.in_channels, self.h_channels, kernel_size=1, stride=1, padding=0,
                                     bias=False)
        self.phi = torch.nn.Conv3d(self.in_channels, self.h_channels, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        self.conv_out = torch.nn.Conv3d(self.h_channels, self.in_channels, kernel_size=1, stride=1, padding=0,
                                        bias=False)

        self.init_weights()

    def init_weights(self, std=0.01):
        for m in [self.g, self.theta, self.phi]:
            torch.nn.init.normal_(m.weight, std=std)
        if self.zeros_init:
            torch.nn.init.constant_(self.conv_out.weight, 0)
        else:
            torch.nn.init.normal_init(self.conv_out.weight, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.h_channels`
            pairwise_weight /= theta_x.shape[-1] ** -0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, t, h, w = x.shape

        # g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.h_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, TxHxW, C]
        theta_x = self.theta(x).view(n, self.h_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, TxHxW]
        phi_x = self.phi(x).view(n, self.h_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, TxHxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.h_channels, t, h, w)

        output = x + self.conv_out(y)

        return output


class SimpleNonLocal3D(torch.nn.Module):
    """ Simplified Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        zeros_init (bool): Zero weight initialization for conv_out
        return_attention (bool): If True, attention map is returned
        extra_args (dict): For specific layer settings (unused)
    """

    def __init__(self, in_channels, reduction=-1, zeros_init=True, return_attention=False, extra_args=None):
        super(SimpleNonLocal3D, self).__init__()
        self.in_channels = in_channels
        self.h_channels = 1 # if reduction == -1 else in_channels // reduction  # TODO Check if anything different than 1 makes sense
        self.zeros_init = zeros_init
        self.return_attention_map = return_attention

        self.phi = torch.nn.Conv3d(self.in_channels, self.h_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = torch.nn.Conv3d(self.in_channels, self.in_channels,
                                        kernel_size=1, stride=1, padding=0, bias=False)

        self.init_weights()

    def init_weights(self, std=0.01):
        torch.nn.init.normal_(self.phi.weight)
        if self.zeros_init:
            torch.nn.init.constant_(self.conv_out.weight, 0)
        else:
            torch.nn.init.normal_(self.conv_out.weight, std)

    def forward(self, x):
        n, c, t, h, w = x.shape

        # x_flat: [N, C, TxHxW]
        x_flat = x.view(n, c, -1)

        # phi_x: [N, TxHxW, 1]
        phi_x = self.phi(x)
        phi_x = phi_x.view(n, -1, self.h_channels)

        # att_map: [N, TxHxW, 1]
        att_map = phi_x.softmax(dim=1)

        # y: [N, C, 1, 1, 1]
        y = torch.matmul(x_flat, att_map)
        y = y.unsqueeze(-1).unsqueeze(-1)

        # Skip connection
        # TODO Review: Global ctx added to current frame only(index: 0).
        z = x[:,:,0:1,:,:] + self.conv_out(y)

        return z.squeeze(2)


class DeformableNonLocal3D(torch.nn.Module):
    """ Deformable Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        zeros_init (bool): True (default) for initializing conv_out to zero
    """

    def __init__(self,
                 in_channels,
                 zeros_init=True, ):
        super(DeformableNonLocal3D, self).__init__()
        self.in_channels = in_channels
        self.zeros_init = zeros_init
        self.h_channels = 1
        self.phi = torch.nn.Conv3d(self.in_channels, self.h_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = torch.nn.Conv3d(self.in_channels, self.in_channels,
                                        kernel_size=1, stride=1, padding=0, bias=False)

        self.init_weights()

    def init_weights(self, std=0.01):
        torch.nn.init.normal_(self.phi.weight)
        if self.zeros_init:
            torch.nn.init.constant_(self.conv_out.weight, 0)
        else:
            torch.nn.init.normal_(self.conv_out.weight)

    def forward(self, x):
        exit('[ERROR] DeformableNonLocal3D not yet implemented')


class DeformableSimpleNonLocal3D(torch.nn.Module):
    """ Deformable Simplified Non-local module.
    See https://arxiv.org/abs/1711.07971 for details.
    Args:
        in_channels (int): Channels of the input feature map.
        zeros_init (bool): True (default) for initializing conv_out to zero
    """

    def __init__(self,
                 in_channels,
                 zeros_init=True, ):
        super(DeformableSimpleNonLocal3D, self).__init__()
        self.in_channels = in_channels
        self.zeros_init = zeros_init
        self.h_channels = 1
        self.phi = torch.nn.Conv3d(self.in_channels, self.h_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = torch.nn.Conv3d(self.in_channels, self.in_channels,
                                        kernel_size=1, stride=1, padding=0, bias=False)

        self.init_weights()

    def init_weights(self, std=0.01):
        torch.nn.init.normal_(self.phi.weight)
        if self.zeros_init:
            torch.nn.init.constant_(self.conv_out.weight, 0)
        else:
            torch.nn.init.normal_(self.conv_out.weight)

    def forward(self, x):
        exit('[ERROR] DeformableSimpleNonLocal3D not yet implemented')


_TRANSFORMATION_MODULES = Registry({
    "BottleneckBN": BottleneckBN,
    "BottleneckWithFixedBatchNorm": BottleneckWithFixedBatchNorm,
    "BottleneckWithGN": BottleneckWithGN,
})

_STEM_MODULES = Registry({
    "StemBN": StemBN,
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
    "StemWithGN": StemWithGN,
})

_NON_LOCAL_CTX_MODULES = Registry({
    "StandardNL": NonLocal3D,
    "SimplifiedNL": SimpleNonLocal3D,
    "RegionNL": RegionNonLocal3D,
    # "DefNL": DeformableNonLocal3D,
    # "DefSNL": DeformableSimpleNonLocal3D,
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
    "R-50-FPN-NL": ResNet50FPNStagesTo5,
})
