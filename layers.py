import torch
import torch.nn as nn
from zonotope import HybridZonotope
from functools import reduce
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.bounds=None

    def report(self):
        assert False

    def reset_bounds(self):
        self.bounds = None


class Conv2d(MyModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dim=None):
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dim = dim
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.reset_parameters()


    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward_concrete(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.conv.padding, self.dilation, self.conv.groups)

    def forward_abstract(self, x):
        return x.conv2d(self.weight, self.bias, self.stride, self.conv.padding, self.dilation, self.conv.groups)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            return self.forward_abstract(x)
        else:
            return self.forward_concrete(x)


class Upsample(MyModule):
    def __init__(self, size, mode="nearest", align_corners=False, consolidate_errors=False):
        super(Upsample, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = None if mode in ["nearest", "area"] else align_corners
        self.upsample = nn.Upsample(size=self.size, mode=self.mode, align_corners=self.align_corners)
        self.consolidate_errors = consolidate_errors

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            x = x.upsample(size=self.size, mode=self.mode, align_corners=self.align_corners, consolidate_errors=self.consolidate_errors)
        else:
            x = self.upsample(x)
        return x


class Sequential(MyModule):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward_between(self, i_from, i_to, x, residual=None):
        for layer in self.layers[i_from:i_to]:
            x = layer(x)
        return x

    def forward_until(self, i, x):
        return self.forward_between(0, i+1, x)

    def forward_from(self, i, x, residual):
        return self.forward_between(i+1, len(self.layers), x, residual)

    def forward(self, x):
        return self.forward_from(-1, x, None)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


class ReLU(MyModule):
    def __init__(self, dims=None):
        super(ReLU, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(-torch.ones(dims, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            out, deepz_lambda = x.relu(self.deepz_lambda, self.bounds)
            if x.domain == "zono_iter" and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        else:
            return x.relu()


class Log(MyModule):
    def __init__(self, dims=None):
        super(Log, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(-torch.ones(dims, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            out, deepz_lambda = x.log(self.deepz_lambda, self.bounds)
            if x.domain == "zono_iter" and (self.deepz_lambda<0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return x.log()


class Exp(MyModule):
    def __init__(self, dims=None):
        super(Exp, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(-torch.ones(dims, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            out, deepz_lambda = x.exp(self.deepz_lambda, self.bounds)
            if x.domain == "zono_iter" and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return x.exp()


class Inv(MyModule):
    def __init__(self, dims=None):
        super(Inv, self).__init__()
        self.dims = dims
        self.deepz_lambda = nn.Parameter(-torch.ones(dims, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            out, deepz_lambda = x.inv(self.deepz_lambda, self.bounds)
            if x.domain == "zono_iter" and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return 1./x


class LogSumExp(MyModule):
    def __init__(self, dims=None):
        super(LogSumExp, self).__init__()
        self.dims = dims
        self.exp = Exp(dims)
        self.log = Log(1)
        self.c = None # for MILP verificiation

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def reset_bounds(self):
        self.log.bounds = None
        self.bounds = None
        self.exp.bounds = None
        self.c = None

    def forward(self, x, bounds=False):
        if isinstance(x, HybridZonotope):
            max_head = x.head.max(dim=1)[0].unsqueeze(1).detach()
            self.c = max_head
            x_temp = x - max_head
            if bounds: add_bounds(None, x_temp, bounds=None, layer=self.exp)
            exp_sum = self.exp(x-max_head).sum(dim=1)
            if bounds: add_bounds(None, exp_sum, bounds=None, layer=self.log)
            log_sum = self.log(exp_sum)
            return log_sum+max_head
        max_head = x.max(dim=1)[0].unsqueeze(1)
        self.c = max_head
        x_tmp = x-max_head
        exp_sum = x_tmp.exp().sum(dim=1).unsqueeze(dim=1)
        log_sum = exp_sum.log() + max_head
        return log_sum


class Entropy(MyModule):
    def __init__(self, dims=None, low_mem=False, neg=False):
        super(Entropy, self).__init__()
        self.dims = dims
        self.exp = Exp(dims)
        self.log_sum_exp = LogSumExp(dims)
        self.low_mem = low_mem
        self.out_features = 1
        self.neg = neg

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def reset_bounds(self):
        self.log_sum_exp.reset_bounds()
        self.bounds=None
        self.exp.bounds=None

    def forward(self, x, bounds=False):
        if isinstance(x, HybridZonotope):
            log_sum = self.log_sum_exp(x, bounds)
            x_temp = x.add(-log_sum, shared_errors=None if x.errors is None else x.errors.size(0))
            if bounds: add_bounds(None, x_temp, bounds=None, layer=self.exp)
            softmax = self.exp(x_temp)
            prob_weighted_act = softmax.prod(x, None if x.errors is None else x.errors.size(0), low_mem=self.low_mem).sum(dim=1)
            entropy = log_sum.add(-prob_weighted_act, shared_errors=None if log_sum.errors is None else log_sum.errors.size(0))
            return entropy * torch.FloatTensor([1-2*self.neg]).to(entropy.head.device)
        log_sum = self.log_sum_exp(x)
        softmax = (x-log_sum).exp()
        prob_weighted_act = (softmax*x).sum(dim=1).unsqueeze(dim=1)
        entropy = log_sum - prob_weighted_act
        return entropy * (1-2*self.neg)


class Flatten(MyModule):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view((x.size()[0], -1))


class Linear(MyModule):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.use_bias = bias
        self.linear = nn.Linear(in_features, out_features, bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            return x.linear(self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


class BatchNorm1d(MyModule):
    def __init__(self, out_features):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            # assert (not self.training)
            return x.batch_norm(self.bn)
        if self.bn.training:
            momentum = 1 if self.bn.momentum is None else self.bn.momentum
            mean = x.mean(dim=[0, 2]).detach()
            var = (x - mean.view(1, -1, 1)).var(unbiased=False, dim=[0, 2]).detach()
            if self.bn.running_mean is not None and self.bn.running_var is not None and self.bn.track_running_stats:
                self.bn.running_mean = self.bn.running_mean * (1 - momentum) + mean * momentum
                self.bn.running_var = self.bn.running_var * (1 - momentum) + var * momentum
            else:
                self.bn.running_mean = mean
                self.bn.running_var = var
        c = (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps))
        b = (-self.bn.running_mean * c + self.bn.bias)
        return x*c.view(1,-1,1)+b.view(1,-1,1)


class BatchNorm2d(MyModule):
    def __init__(self, out_features):
        super(BatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(out_features, momentum=0.1)
        self.current_mean = None
        self.current_var = None

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            return x.batch_norm(self.bn, self.current_mean.clone(), self.current_var.clone())
        if self.bn.training:
            momentum = 1 if self.bn.momentum is None else self.bn.momentum
            self.current_mean = x.mean(dim=[0, 2, 3]).detach()
            self.current_var = x.var(unbiased=False, dim=[0, 2, 3]).detach()
            if self.bn.track_running_stats:
                if self.bn.running_mean is not None and self.bn.running_var is not None:
                    self.bn.running_mean = self.bn.running_mean * (1 - momentum) + self.current_mean * momentum
                    self.bn.running_var = self.bn.running_var * (1 - momentum) + self.current_var * momentum
                else:
                    self.bn.running_mean = self.current_mean
                    self.bn.running_var = self.current_var
        else:
            self.current_mean = self.bn.running_mean
            self.current_var = self.bn.running_var

        c = (self.bn.weight / torch.sqrt(self.current_var + self.bn.eps))
        b = (-self.current_mean * c + self.bn.bias)
        return x*c.view(1,-1,1,1)+b.view(1,-1,1,1)


class AvgPool2d(MyModule):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.avg_pool2d = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            assert self.padding == 0
            return x.avg_pool2d(self.kernel_size, self.stride)
        return self.avg_pool2d(x)


class MaxPool2d(MyModule):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_pool2d = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            assert self.padding == 0
            return x.max_pool2d(self.kernel_size, self.stride)
        return self.max_pool2d(x)


class GlobalAvgPool2d(MyModule):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        self.size = 1
        self.global_avg_pool2d = nn.AdaptiveAvgPool2d(self.size)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            return x.global_avg_pool2d()
        return self.global_avg_pool2d(x)


class Bias(MyModule):
    def __init__(self, bias=0, fixed=False):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(bias*torch.ones(1))
        self.bias.requires_grad_(not fixed)

    def forward(self, x):
        return x + self.bias


class Scale(MyModule):
    def __init__(self, scale=1, fixed=False):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(scale*torch.ones(1))
        self.scale.requires_grad_(not fixed)

    def forward(self, x):
        return x * self.scale


class Normalization(MyModule):
    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean)
        self.sigma = nn.Parameter(sigma)
        self.mean.requires_grad_(False)
        self.sigma.requires_grad_(False)

    def forward(self, x):
        if isinstance(x, HybridZonotope):
            return x.normalize(self.mean, self.sigma)
        return (x - self.mean) / self.sigma


class PreActBlock(MyModule):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(PreActBlock, self).__init__()
        self.path1 = Sequential(
            BatchNorm2d(in_planes),
            ReLU((in_planes, dim, dim)),
            Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, dim=dim),
            BatchNorm2d(planes),
            ReLU((planes, dim//stride, dim//stride)),
            Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, dim=dim//stride),
        )
        self.path2 = Sequential()
        if in_planes != planes:
            self.path2 = Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, dim=dim)

    def forward(self, x):
        out = self.path1(x)
        y = self.path2(x)
        out += y
        return out


class ResBlock(MyModule):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, mode="standard"):
        super(ResBlock, self).__init__()

        self.residual = self.get_residual_layers( mode, in_planes, planes, stride, dim)
        self.downsample = downsample
        self.relu_final = ReLU((planes, dim//stride, dim//stride)) if mode=="standard" else None

    def forward(self, x):
        identity = x
        out = self.residual(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        if isinstance(out, HybridZonotope):
            out = out.add(identity, shared_errors=None if identity.errors is None else identity.errors.size(0))
        else:
            out += identity
        if self.relu_final is not None:
            out = self.relu_final(out)
        return out

    def get_residual_layers(self, mode, in_planes, out_planes, stride, dim):
        if mode == "standard":
            residual = Sequential(
                Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim // stride, dim // stride)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d(out_planes),
            )
        elif mode == "wide":
            residual = Sequential(
                BatchNorm2d(in_planes),
                ReLU((in_planes, dim, dim)),
                Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim, dim)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
            )
        elif mode == "fixup":
            residual = Sequential(
                Bias(),
                Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                Bias(),
                ReLU((out_planes, dim // stride, dim // stride)),
                Bias(),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                Scale(),
                Bias(),
            )
        else:
            raise RuntimeError(f"Unknown layer mode {mode:%s}")

        return residual


class BasicBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="standard")


class WideBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(WideBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="wide")


class FixupBasicBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="fixup")

        if downsample is not None:
            self.downsample = Sequential(
                self.residual.layers[0],
                downsample)


def add_bounds(lidx, zono, bounds=None, layer=None):
    lb_new, ub_new = zono.concretize()
    if layer is not None:
        if layer.bounds is not None:
            lb_old, ub_old = layer.bounds
            lb_new, ub_new = torch.max(lb_old, lb_new).detach(), torch.min(ub_old, ub_new).detach()
        layer.bounds = (lb_new, ub_new)
    if bounds is not None:
        bounds[lidx] = (lb_new, ub_new)
        return bounds