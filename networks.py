import numpy as np
import torch.nn as nn
import torch
from warnings import warn
from zonotope import HybridZonotope
from loaders import get_mean_sigma
from efficientnet_pytorch import EfficientNet as ENet
from layers import (Conv2d, Normalization, ReLU, Flatten, Linear, Sequential, PreActBlock, BasicBlock, FixupBasicBlock, Bias, Scale,
    BatchNorm1d, BatchNorm2d, AvgPool2d, Entropy, GlobalAvgPool2d, Upsample, WideBlock, MaxPool2d)


class UpscaleNet(nn.Module):
    def __init__(self, device, dataset, adv_pre, input_size, net, net_dim):
        super(UpscaleNet, self).__init__()
        self.net = net
        self.net_dim = net_dim
        self.blocks = []

        if input_size == net_dim:
            self.transform = None
        else:
            self.transform = Upsample(size=self.net_dim, mode="nearest", align_corners=False,consolidate_errors=False)
            self.blocks += [self.transform]

        if adv_pre:
            self.blocks += [Scale(2, fixed=True), Bias(-1, fixed=True)]
            self.normalization = Sequential(*self.blocks)
        else:
            mean, sigma = get_mean_sigma(device, dataset)
            self.normalization = Normalization(mean, sigma)
            self.blocks += [self.normalization]

        self.blocks += [self.net]

    def forward(self, x, residual=None, input_idx=-1):
        assert residual is None and input_idx == -1
        for block in self.blocks:
            x = block(x)
        return x

    def forward_between(self, i_from, i_to, x):
        if not (i_from == 0 and (i_to == len(self.blocks) or i_to is None)):
            raise NotImplementedError("Partial propagation for UpscaleNets is not implemented")
        x = self.forward(x)
        return x

    def freeze(self, layer_idx):
        if layer_idx <= 0:
            pass
        elif layer_idx == len(self.blocks) - 1:
            for n_param, param in self.named_parameters():
                param.requires_grad_(False)
        else:
            raise NotImplementedError("Freezing of UpscaleNets is not implemented")

    def reset_bounds(self):
        # raise NotImplementedError("Resetting Bounds of UpscaleNets is not implemented")
        warn("UpscaleNets do not support bound computation. No bounds reset.")
        pass

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def determine_dims(self, x, force=False, blocks=None):
        self.dims = None


class SeqNet(nn.Module):

    def __init__(self, blocks=None, net_dim=None):
        super(SeqNet, self).__init__()
        self.is_double = False
        self.dims = []
        self.net_dim = net_dim
        self.transform = None if net_dim is None else Upsample(size=self.net_dim, mode="bilinear", consolidate_errors=True)
        self.blocks = [] if self.transform is None else [self.transform]
        if blocks is not None:
            self.blocks += [*blocks]
            self.blocks = Sequential(*self.blocks)

    def forward(self, x, residual=None, input_idx=-1):
        if isinstance(x, torch.Tensor) and self.is_double:
            x = x.to(dtype=torch.float64)
        x = self.forward_between(input_idx+1, None, x, residual)
        return x

    def verify(self, inputs, targets, eps, domain, threshold_min=0, input_min=0, input_max=1, return_abs=False):
        n_class = self.blocks[-1].out_features
        device = inputs.device

        if self.transform is not None and self.transform.consolidate_errors:
            abs_input = HybridZonotope.construct_from_noise(inputs, eps, "box",
                                                            data_range=(input_min, input_max))
            abs_input.domain = domain
        else:
            abs_input = HybridZonotope.construct_from_noise(inputs, eps, domain,
                                                            data_range=(input_min, input_max))

        if domain in ["box","hbox"] and n_class > 1:
            C = torch.stack([self.get_c_mat(n_class, x, device) for x in targets], dim=0)
            I = (~(targets.unsqueeze(1) == torch.arange(n_class, dtype=torch.float32, device=device).unsqueeze(0)))
            abs_outputs = self.forward_between(0, len(self.blocks) - 1, abs_input)
            abs_outputs = abs_outputs.linear(self.blocks[-1].linear.weight, self.blocks[-1].linear.bias, C)
            threshold_n = abs_outputs.concretize()[0][I].view(targets.size(0), n_class - 1).min(dim=1)[0]
            ver_corr = threshold_n > threshold_min
            ver = ver_corr
        else:
            abs_outputs = self(abs_input)
            ver, ver_corr, threshold_n = abs_outputs.verify(targets, threshold_min=threshold_min,
                                                            corr_only=abs_outputs.size(-1) > 10)

        if return_abs:
            return ver, ver_corr, threshold_n, abs_outputs
        else:
            return ver, ver_corr, threshold_n

    @staticmethod
    def get_c_mat(n_class, target, device):
        c = torch.eye(n_class, dtype=torch.float32, device=device)[target].unsqueeze(dim=0) \
            - torch.eye(n_class, dtype=torch.float32, device=device)
        return c

    def freeze(self, layer_idx):
        for i in range(layer_idx+1):
            self.blocks[i].requires_grad_(False)
            if isinstance(self.blocks[i],BatchNorm1d) or isinstance(self.blocks[i],BatchNorm2d):
                self.blocks[i].training = False

    def reset_bounds(self):
        for block in self.blocks:
            block.reset_bounds()

    def to_double(self):
        self.is_double = True
        for param_name, param_value in self.named_parameters():
            param_value.data = param_value.data.to(dtype=torch.float64)

    def forward_between(self, i_from, i_to, x, residual=None):
        """ Forward from (inclusive) to (exclusive)"""
        if i_to is None:
            i_to = len(self.blocks)
        if i_from is None:
            i_from = 0
        x = self.blocks.forward_between(i_from, i_to, x, residual=residual)
        return x

    def forward_until(self, i, x):
        """ Forward until layer i (inclusive) """
        x = self.forward_between(None, i+1, x, residual=None)
        return x

    def forward_from(self, i, x):
        """ Forward from layer i (exclusive) """
        x = self.forward_between(i+1, None, x, residual=None)
        return x

    def temp_freeze(self):
        param_state = {}
        for name, param in self.named_parameters():
            param_state[name] = param.requires_grad
            param.requires_grad = False
        return param_state

    def get_freeze_state(self):
        param_state = {}
        for name, param in self.named_parameters():
            param_state[name] = param.requires_grad
        return param_state

    def restore_freeze(self, param_state):
        for name, param in self.named_parameters():
            param.requires_grad = param_state[name]

    def determine_dims(self, x, force=False, blocks=None):
        if len(self.dims)>0 and not force:
            return
        if blocks is None:
            blocks = self.blocks
        for layer in blocks:
            if hasattr(layer, "layers"):
                for sub_layers in layer.layers:
                    sub_layers = sub_layers if not hasattr(sub_layers, "residual") else sub_layers.residual
                    x = self.determine_dims(x, force=True, blocks=sub_layers)
            else:
                x = layer(x)
            self.dims += [tuple(x.size()[1:])]
        return x

    def get_subNet_blocks(self,startBlock=0, endBlock=None):
        if endBlock is None:
            endBlock=len(self.blocks)
        assert endBlock<=len(self.blocks)
        return self.blocks[startBlock:endBlock]


class myNet(SeqNet):
    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, conv_widths=None,
                 kernel_sizes=None, linear_sizes=None, depth_conv=None, paddings=None, strides=None,
                 dilations=None, pool=False, net_dim=None, bn=False, max=False, scale_width=True):
        super(myNet, self).__init__(net_dim=None if net_dim == input_size else net_dim)
        if kernel_sizes is None:
            kernel_sizes = [3]
        if conv_widths is None:
            conv_widths = [2]
        if linear_sizes is None:
            linear_sizes = [200]
        if paddings is None:
            paddings = [1]
        if strides is None:
            strides = [2]
        if dilations is None:
            dilations = [1]
        if net_dim is None:
            net_dim = input_size

        if len(conv_widths) != len(kernel_sizes):
            kernel_sizes = len(conv_widths) * [kernel_sizes[0]]
        if len(conv_widths) != len(paddings):
            paddings = len(conv_widths) * [paddings[0]]
        if len(conv_widths) != len(strides):
            strides = len(conv_widths) * [strides[0]]
        if len(conv_widths) != len(dilations):
            dilations = len(conv_widths) * [dilations[0]]

        self.n_class=n_class
        self.input_size=input_size
        self.input_channel=input_channel
        self.conv_widths=conv_widths
        self.kernel_sizes=kernel_sizes
        self.paddings=paddings
        self.strides=strides
        self.dilations = dilations
        self.linear_sizes=linear_sizes
        self.depth_conv=depth_conv
        self.net_dim = net_dim
        self.bn=bn
        self.max=max

        mean, sigma = get_mean_sigma(device, dataset)
        layers = self.blocks
        layers += [Normalization(mean, sigma)]

        N = net_dim
        n_channels = input_channel
        self.dims += [(n_channels,N,N)]

        for width, kernel_size, padding, stride, dilation in zip(conv_widths, kernel_sizes, paddings, strides, dilations):
            if scale_width:
                width *= 16
            N = int(np.floor((N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            layers += [Conv2d(n_channels, int(width), kernel_size, stride=stride, padding=padding, dilation=dilation)]
            if self.bn:
                layers += [BatchNorm2d(int(width))]
            if self.max:
                layers += [MaxPool2d(int(width))]
            layers += [ReLU((int(width), N, N))]
            n_channels = int(width)
            self.dims += 2*[(n_channels,N,N)]


        if depth_conv is not None:
            layers += [Conv2d(n_channels, depth_conv, 1, stride=1, padding=0),
                       ReLU((n_channels, N, N))]
            n_channels = depth_conv
            self.dims += 2*[(n_channels,N,N)]

        if pool:
            layers += [GlobalAvgPool2d()]
            self.dims += 2 * [(n_channels, 1, 1)]
            N=1

        layers += [Flatten()]
        N = n_channels * N ** 2
        self.dims += [(N,)]


        for width in linear_sizes:
            if width == 0:
                continue
            layers += [Linear(int(N), int(width)),
                       ReLU(width)]
            N = width
            self.dims+=2*[(N,)]

        layers += [Linear(N, n_class)]
        self.dims+=[(n_class,)]

        self.blocks = Sequential(*layers)


class FFNN(myNet):
    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3, net_dim=None):
        super(FFNN,self).__init__(device, dataset, n_class, input_size, input_channel, conv_widths=[], linear_sizes=sizes, net_dim=net_dim)


class ConvMedBig(myNet):
    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, width1=1, width2=1, width3=1, linear_size=100, net_dim=None):
        super(ConvMedBig, self).__init__(device, dataset, n_class, input_size, input_channel,
                                         conv_widths=[width1,width2,2*width3], kernel_sizes=[3,4,4],
                                         linear_sizes=[linear_size], strides=[1,2,2], net_dim=net_dim)


class MyResnet(SeqNet):
    def __init__(self, device, dataset, n_blocks, n_class=10, input_size=32, input_channel=3, block='basic',
                 in_planes=32, net_dim=None, widen_factor=1, pooling="global"):
        super(MyResnet, self).__init__(net_dim=None if net_dim == input_size else net_dim)
        if block == 'basic':
            self.res_block = BasicBlock
        elif block == 'preact':
            self.res_block = PreActBlock
        elif block == 'wide':
            self.res_block = WideBlock
        elif block == 'fixup':
            self.res_block = FixupBasicBlock
        else:
            assert False
        self.n_layers = sum(n_blocks)
        mean, sigma = get_mean_sigma(device, dataset)
        dim = input_size

        k = widen_factor

        layers = [Normalization(mean, sigma),
                  Conv2d(input_channel, in_planes, kernel_size=3, stride=1, padding=1, bias=(block == "wide"), dim=dim)]

        if not block == "wide":
            layers += [Bias() if block == 'fixup' else BatchNorm2d(in_planes),
                       ReLU((in_planes, input_size, input_size))]

        strides = [1, 2] + ([2] if len(n_blocks) > 2 else []) + [1] * max(0,(len(n_blocks)-3))

        n_filters = in_planes
        for n_block, n_stride in zip(n_blocks, strides):
            if n_stride > 1:
                n_filters *= 2
            dim, block_layers = self.get_block(in_planes, n_filters*k, n_block, n_stride, dim=dim)
            in_planes = n_filters*k
            layers += [block_layers]

        if block == 'fixup':
            layers += [Bias()]
        else:
            layers += [BatchNorm2d(n_filters*k)]

        if block == "wide":
            layers += [ReLU((n_filters*k, dim, dim))]

        if pooling == "global":
            layers += [GlobalAvgPool2d()]
            N = n_filters * k
        elif pooling == "None":      # old networks miss pooling layer and wont load
            N = n_filters * dim * dim * k
        elif isinstance(pooling, int):
            layers += [AvgPool2d(pooling)]
            dim = dim//pooling
            N = n_filters * dim * dim * k

        layers += [Flatten(), ReLU(N)]

        if block == 'fixup':
            layers += [Bias()]

        layers += [Linear(N, n_class)]

        self.blocks = Sequential(*layers)

        # Fixup initialization
        if block == 'fixup':
            for m in self.modules():
                if isinstance(m, FixupBasicBlock):
                    conv1, conv2 = m.residual[1].conv, m.residual[5].conv
                    nn.init.normal_(conv1.weight,
                                    mean=0,
                                    std=np.sqrt(2 / (conv1.weight.shape[0] * np.prod(conv1.weight.shape[2:]))) * self.n_layers ** (-0.5))
                    nn.init.constant_(conv2.weight, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def get_block(self, in_planes, out_planes, n_blocks, stride, dim):
        strides = [stride] + [1]*(n_blocks-1)
        layers = []

        if in_planes != out_planes:
            if self.res_block == FixupBasicBlock:
                downsample = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                # downsample = AvgPool2d(1, stride=stride)
            elif self.res_block == WideBlock:
                downsample = Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)
            else:
                downsample = [Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)]
                # downsample = [AvgPool2d(1, stride=stride)]
                downsample += [BatchNorm2d(out_planes)]
                downsample = Sequential(*downsample)

        for stride in strides:
            layers += [self.res_block(dim, in_planes, out_planes, stride, downsample)]
            downsample = None
            in_planes = out_planes
            dim = dim // stride
        return dim, Sequential(*layers)


class EfficientNet(UpscaleNet):
    def __init__(self, device, dataset, name, input_dim, in_channels, n_classes, pretrained=False, adv=False):
        if pretrained:
            net = ENet.from_pretrained(name, in_channels=in_channels, num_classes=n_classes, advprop=adv)
        else:
            net = ENet.from_name(name, in_channels=in_channels, num_classes=n_classes)
        super(EfficientNet, self).__init__(device, dataset, adv, input_dim, net, net._global_params.image_size)