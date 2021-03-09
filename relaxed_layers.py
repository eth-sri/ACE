import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import my_cauchy


class RelaxedLayer(nn.Module):
    def __init__(self, layer=None):
        super(RelaxedLayer,self).__init__()
        self.layer = layer
        self.bounds = None if layer is None else layer.bounds


class RelaxedConv2d(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedConv2d, self).__init__(layer)
        self.conv = self.layer.conv

    def forward(self, curr_head, curr_errors):
        curr_head = F.conv2d(curr_head, self.layer.weight, self.layer.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        curr_errors = F.conv2d(curr_errors, self.layer.weight, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return curr_head, curr_errors


class RelaxedLinear(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedLinear, self).__init__(layer)
        self.linear = self.layer.linear

    def forward(self, curr_head, curr_errors):
        curr_head = F.linear(curr_head, self.layer.weight, self.layer.bias)
        curr_errors = F.linear(curr_errors, self.layer.weight, bias=None)
        return curr_head, curr_errors


class RelaxedBias(RelaxedLayer):
    def __init__(self,layer):
        super(RelaxedBias, self).__init__(layer)
        self.bias = self.layer.bias

    def forward(self, curr_head, curr_errors):
        curr_head = curr_head + self.layer.bias
        return curr_head, curr_errors


class RelaxedScale(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedScale, self).__init__(layer)
        self.scale = self.layer.scale

    def forward(self, curr_head, curr_errors):
        curr_head = curr_head*self.scale
        curr_errors = curr_errors*self.scale
        return curr_head, curr_errors


class RelaxedNorm(RelaxedLayer):

    def __init__(self, layer):
        super(RelaxedNorm, self).__init__(layer)
        self.mean = self.layer.mean
        self.sigma = self.layer.sigma

    def forward(self, curr_head, curr_errors):
        curr_head = (curr_head - self.mean) / self.sigma
        curr_errors /= self.sigma
        return curr_head, curr_errors

class RelaxedUpsample(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedUpsample, self).__init__(layer)
        self.upsample = self.layer.upsample

    def forward(self, curr_head, curr_errors):
        curr_head = self.upsample(curr_head)
        curr_errors = self.upsample(curr_errors)
        return curr_head, curr_errors


class RelaxedBatchNorm2d(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedBatchNorm2d, self).__init__(layer)
        self.bn = self.layer.bn

    def forward(self, curr_head, curr_errors):
        if self.bn.training:
            momentum = 1 if self.bn.momentum is None else self.bn.momentum
            mean = curr_head.mean(dim=[0, 2, 3]).detach()
            var = (curr_head - mean.view(1, -1, 1, 1)).var(unbiased=False, dim=[0, 2, 3]).detach()
            if self.bn.running_mean is not None and self.bn.running_var is not None and self.bn.track_running_stats:
                self.bn.running_mean = self.bn.running_mean * (1 - momentum) + mean * momentum
                self.bn.running_var = self.bn.running_var * (1 - momentum) + var * momentum
            else:
                self.bn.running_mean = mean
                self.bn.running_var = var
        c = (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps))
        b = (-self.bn.running_mean * c + self.bn.bias)
        curr_head = curr_head * c.view(1,-1,1,1) + b.view(1,-1,1,1)
        curr_errors = curr_errors*c.view(1,-1,1,1)
        return curr_head, curr_errors


class RelaxedBatchNorm1d(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedBatchNorm1d, self).__init__(layer)
        self.bn = self.layer.bn

    def forward(self, curr_head, curr_errors):
        curr_head = self.bn(curr_head)
        curr_errors = curr_errors * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)).view(1, -1, 1)
        return curr_head, curr_errors


class RelaxedResidualBlockStart(RelaxedLayer):
    def __init__(self,layer):
        super(RelaxedResidualBlockStart, self).__init__(layer)

    def forward(self, curr_head, curr_errors):
        return curr_head, curr_errors


class RelaxedReLU(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedReLU, self).__init__(layer)
        self.deepz_lambda = self.layer.deepz_lambda

    def forward(self, curr_head, curr_errors, adv_errors=None, k=None, zono_iter=False):
        D = 1e-9
        lb, ub = self.bounds
        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:])
        is_cross = (lb < 0) & (ub > 0)

        if self.deepz_lambda is not None and zono_iter:
            assert (self.deepz_lambda >= 0).all() and (self.deepz_lambda <= 1).all()
            relu_lambda = self.deepz_lambda
            relu_mu = torch.where(relu_lambda < ub / (ub - lb + D), 0.5 * ub * (1 - relu_lambda),
                        - 0.5 * relu_lambda * lb)
        else:
            relu_lambda = ub / (ub - lb + D)
            relu_mu =- 0.5*relu_lambda*lb
        relu_lambda_cross = torch.where(is_cross, relu_lambda, (lb >= 0).float())
        relu_mu_cross = torch.where(is_cross, relu_mu, torch.zeros_like(relu_mu))

        curr_head = curr_head * relu_lambda_cross + relu_mu_cross
        curr_errors = curr_errors * relu_lambda_cross.unsqueeze(1)

        if k is not None:
            new_errors = relu_mu_cross.unsqueeze(1) * my_cauchy(curr_head.shape[0], k, *curr_head.shape[1:]).to(curr_head.device)
        else:
            new_errors = relu_mu_cross.unsqueeze(1) * adv_errors
        curr_errors = torch.cat([curr_errors, new_errors], dim=1)
        curr_errors = curr_errors.view(-1, *curr_head.shape[1:])
        return curr_head, curr_errors


class RelaxedEntropy(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedEntropy, self).__init__(layer)
        self.exp = RelaxedExp(layer.exp)
        self.log_sum_exp = RelaxedLogSumExp(layer.log_sum_exp)

    def forward(self, curr_head, curr_errors, is_cauchy, adv_errors=None, k=None, zono_iter=False, adv_error_pre=None):
        ls_head, ls_errors, adv_errors = self.log_sum_exp(curr_head, curr_errors, is_cauchy, adv_errors, k, zono_iter, adv_error_pre)

        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:])
        sum_head, sum_errors = RelaxedAdd.forward(curr_head, curr_errors, -ls_head, -ls_errors, 0 if curr_errors is None else curr_errors.size(0))

        if is_cauchy:
            self.layer.exp.bounds = bounds_from_cauchy(sum_head, sum_errors, k)
            softmax_head, softmax_errors = self.exp(sum_head, sum_errors, k=k, zono_iter=zono_iter)
        else:
            adv_error_id = "softmax" if adv_error_pre is None else "%s_softmax" % (adv_error_pre)
            if adv_error_id not in adv_errors:
                adv_errors[adv_error_id] = torch.FloatTensor(curr_head.shape).unsqueeze(1).to(curr_head.device).uniform_(-1, 1).requires_grad_(True)
            softmax_head, softmax_errors = self.exp(sum_head, sum_errors, adv_errors=adv_errors[adv_error_id], zono_iter=zono_iter)

        if is_cauchy:
            prob_head, prob_errors = RelaxedProd.forward(curr_head, curr_errors, softmax_head, softmax_errors, k=k, zono_iter=zono_iter)
        else:
            adv_error_id = "prod" if adv_error_pre is None else "%s_prod" % (adv_error_pre)
            if adv_error_id not in adv_errors:
                adv_errors[adv_error_id] = torch.FloatTensor(curr_head.shape).unsqueeze(1).to(curr_head.device).uniform_(-1, 1).requires_grad_(True)
            prob_head, prob_errors = RelaxedProd.forward(curr_head, curr_errors, softmax_head, softmax_errors,
                                                         adv_errors=adv_errors[adv_error_id], zono_iter=zono_iter)

        prob_head, prob_errors = prob_head.sum(dim=1), prob_errors.sum(dim=2)

        entropy_head, entropy_errors = RelaxedAdd.forward(ls_head, ls_head, -prob_head, -prob_errors, 0 if ls_errors is None else ls_errors.size(0))
        entropy_errors = entropy_errors.view(-1, *entropy_head.shape[1:])
        return entropy_head * (1-2*self.layer.neg), entropy_errors * (1-2*self.layer.neg)


class RelaxedLogSumExp(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedLogSumExp, self).__init__(layer)
        self.exp = RelaxedExp(layer.exp)
        self.log = RelaxedLog(layer.log)

    def forward(self, curr_head, curr_errors, is_cauchy, adv_errors=None, k=None, zono_iter=False, adv_error_pre=None):
        max_head = curr_head.max(dim=1)[0].unsqueeze(1).detach()
        temp_head = curr_head-max_head
        temp_errors = curr_errors

        if is_cauchy:
            self.layer.exp.bounds = bounds_from_cauchy(temp_head, temp_errors, k)
            exp_head, exp_errors = self.exp(temp_head, temp_errors, k=k, zono_iter=zono_iter)
        else:
            adv_error_id = "lse_exp" if adv_error_pre is None else "%s_lse_exp" % (adv_error_pre)
            if adv_error_id not in adv_errors:
                adv_errors[adv_error_id] = torch.FloatTensor(curr_head.shape).unsqueeze(1).to(
                    curr_head.device).uniform_(-1, 1).requires_grad_(True)
            exp_head, exp_errors = self.exp(temp_head, temp_errors, adv_errors=adv_errors[adv_error_id], zono_iter=zono_iter)

        exp_errors = exp_errors.view(exp_head.shape[0], -1, *exp_head.shape[1:])
        exp_head, exp_errors = exp_head.sum(dim=1), exp_errors.sum(dim=2)

        if is_cauchy:
            self.layer.log.bounds = bounds_from_cauchy(exp_head, exp_errors, k)
            log_sum_head, log_sum_errors = self.log(exp_head, exp_errors, k=k, zono_iter=zono_iter)
        else:
            adv_error_id = "lse_log" if adv_error_pre is None else "%s_lse_log" % (adv_error_pre)
            if adv_error_id not in adv_errors:
                adv_errors[adv_error_id] = torch.FloatTensor(curr_head.shape).unsqueeze(1).to(
                    curr_head.device).uniform_(-1, 1).requires_grad_(True)
            log_sum_head, log_sum_errors = self.log(exp_head, exp_errors, adv_errors=adv_errors[adv_error_id],
                                                    zono_iter=zono_iter)
        log_sum_head += max_head
        log_sum_errors = log_sum_errors.view(-1, *log_sum_errors.shape[1:])
        return log_sum_head, log_sum_errors


class RelaxedExp(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedExp, self).__init__(layer)

    def forward(self, curr_head, curr_errors, adv_errors=None, k=None, zono_iter=False):
        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:])
        lb, ub = self.layer.bounds
        D = 1e-6
        is_tight = (ub == lb)
        exp_lambda_s = (ub.exp() - lb.exp()) / (ub - lb + D)
        exp_lambda = exp_lambda_s
        exp_lambda = torch.min(exp_lambda, (lb + 1 - 1e-1).exp())  # ensure non-negative output only

        exp_ub_0 = torch.where(exp_lambda > exp_lambda_s, lb.exp() - exp_lambda * lb, ub.exp() - exp_lambda * ub)
        exp_mu = 0.5 * (exp_lambda * (1 - exp_lambda.log()) + exp_ub_0)
        exp_delta = 0.5 * (exp_ub_0 - exp_lambda * (1 - exp_lambda.log()))

        exp_lambda = torch.where(is_tight, torch.zeros_like(exp_lambda), exp_lambda)
        exp_mu = torch.where(is_tight, ub.exp(), exp_mu)
        exp_delta = torch.where(is_tight, torch.zeros_like(exp_delta), exp_delta)

        assert (not torch.isnan(exp_mu).any()) and (not torch.isnan(exp_lambda).any())

        new_head = curr_head * exp_lambda + exp_mu
        errors = curr_errors * exp_lambda.unsqueeze(dim=1)
        curr_errors = curr_errors.view(-1, *curr_head.shape[1:])

        if k is not None:
            new_errors = torch.cat([errors, exp_delta.unsqueeze(1) * my_cauchy(curr_head.shape[0], k, *curr_head.shape[1:]).to(curr_head.device)], dim=1)
        else:
            new_errors = torch.cat([errors, exp_delta.unsqueeze(1) * adv_errors], dim=1)
        new_errors = new_errors.view(-1, *new_head.shape[1:])
        return new_head , new_errors


class RelaxedLog(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedLog, self).__init__(layer)
        self.deepz_lambda = self.layer.deepz_lambda

    def forward(self, curr_head, curr_errors, adv_errors=None, k=None, zono_iter=False):
        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:])
        D = 1e-6
        lb, ub = self.layer.bounds

        is_tight = (ub == lb)

        log_lambda_s = (ub.log() - (lb + D).log()) / (ub - lb + D)
        log_lambda = log_lambda_s

        log_lb_0 = torch.where(log_lambda < log_lambda_s, ub.log() - ub * log_lambda, (lb + D).log() - lb * log_lambda)
        log_mu = 0.5 * (-log_lambda.log() - 1 + log_lb_0)
        log_delta = 0.5 * (-log_lambda.log() - 1 - log_lb_0)

        log_lambda = torch.where(is_tight, torch.zeros_like(log_lambda), log_lambda)
        log_mu = torch.where(is_tight, ub.log(), log_mu)
        log_delta = torch.where(is_tight, torch.zeros_like(log_delta), log_delta)

        assert (not torch.isnan(log_mu).any()) and (not torch.isnan(log_lambda).any())

        new_head = curr_head * log_lambda + log_mu
        errors = curr_errors * log_lambda.unsqueeze(dim=1)
        curr_errors = curr_errors.view(-1,*curr_head.shape[1:])

        if k is not None:
            new_errors = torch.cat([errors, log_delta.unsqueeze(1) * my_cauchy(curr_head.shape[0], k, *curr_head.shape[1:]).to(curr_head.device)], dim=1)
        else:
            new_errors = torch.cat([errors, log_delta.unsqueeze(1) * adv_errors], dim=1)
        new_errors = new_errors.view(-1, *new_head.shape[1:])
        return new_head, new_errors

class RelaxedProd(RelaxedLayer):
    @staticmethod
    def forward(self, curr_head, curr_errors, curr_head_2, curr_errors_2, shared_errors, adv_errors=None, k=None, zono_iter=False):
        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:]).transpose(1,0)
        curr_errors_2 = curr_errors_2.view(curr_head_2.shape[0], -1, *curr_head_2.shape[1:]).transpose(1,0)

        assert all([x == y or x == 1 or y == 1 for x, y in zip(curr_head.shape[::-1], curr_head_2.shape[::-1])])
        if shared_errors is None:
            shared_errors = 0
        if shared_errors == -1:
            shared_errors = curr_errors.size(0) if curr_errors.size(0) == curr_errors_2.size(0) else 0

        self_errors = torch.cat([curr_errors
                                 * torch.ones((curr_errors.size(0),) + tuple(curr_head_2.shape),
                                              dtype=curr_errors.dtype, device=curr_errors.device),
                                 torch.zeros((curr_errors_2.size(0) - shared_errors,) + tuple(curr_head.shape),
                                             dtype=curr_errors.dtype, device=curr_errors.device)
                                 * torch.ones((curr_errors_2.size(0) - shared_errors,) + tuple(curr_head_2.shape),
                                              dtype=curr_errors.dtype, device=curr_errors.device)], dim=0)
        factor_errors = torch.cat([curr_errors_2[:shared_errors]
                                   * torch.ones_like(curr_errors[:shared_errors]),
                                   torch.zeros((curr_errors.size(0) - shared_errors,) + tuple(curr_head.shape),
                                               dtype=curr_errors.dtype, device=curr_errors.device)
                                   * torch.ones((curr_errors.size(0) - shared_errors,) + tuple(curr_head_2.shape),
                                                dtype=curr_errors.dtype, device=curr_errors.device),
                                   curr_errors_2[shared_errors:]
                                   * torch.ones((curr_errors_2.size(0) - shared_errors,) + tuple(curr_head.shape),
                                                dtype=curr_errors.dtype, device=curr_errors.device)], dim=0)

        lin_err = curr_head.unsqueeze(dim=0) * factor_errors + curr_head_2.unsqueeze(dim=0) * self_errors
        quadr_const = (self_errors * factor_errors)
        quadr_error_tmp = self_errors.unsqueeze(1) * self_errors.unsqueeze(0)
        quadr_error_tmp = 1. / 2. * (quadr_error_tmp + quadr_error_tmp.transpose(1, 0)).abs().sum(dim=1).sum(dim=0) \
                          - 1 / 2 * quadr_const.abs().sum(dim=0)

        new_head = curr_head * curr_head_2 + 1 / 2 * quadr_const.sum(dim=0)
        errors = lin_err.transpose(1,0)
        curr_errors = curr_errors.transpose(1,0).view(-1, *curr_head.shape[1:])
        curr_errors_2 = curr_errors_2.transpose(1,0).view(-1, *curr_head_2.shape[1:])

        if k is not None:
            new_errors = torch.cat([errors, quadr_error_tmp.unsqueeze(1) *
                                    my_cauchy(curr_head.shape[0], k, *curr_head.shape[1:]).to(curr_head.device)], dim=1)
        else:
            new_errors = torch.cat([errors, quadr_error_tmp.unsqueeze(1) * adv_errors], dim=1)
        new_errors = new_errors.view(-1, *new_head.shape[1:])
        return new_head, new_errors


class RelaxedAdd(RelaxedLayer):
    def __init__(self):
        super(RelaxedAdd,self).__init__()

    @staticmethod
    def forward(self, curr_head, curr_errors, curr_head_2, curr_errors_2, shared_errors):
        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:]).transpose(1,0)
        curr_errors_2 = curr_errors_2.view(curr_head_2.shape[0], -1, *curr_head_2.shape[1:]).transpose(1,0)

        assert all([x == y or x == 1 or y == 1 for x, y in zip(curr_head.shape[::-1], curr_head_2.shape[::-1])])

        new_head = curr_head + curr_head_2

        if shared_errors < 0:
            shared_errors = curr_errors.size(0) if curr_errors.size(0) == curr_errors_2.size(0) else 0

        if curr_errors is None:
            new_errors = curr_errors_2
        elif curr_errors_2 is None:
            new_errors = curr_errors
        else:
            self_errors = torch.cat([curr_errors
                                     * torch.ones((curr_errors.size(0),) + tuple(curr_head_2.shape),
                                                  dtype=curr_errors.dtype, device=curr_errors.device),
                                     torch.zeros(
                                         (curr_errors_2.size(0) - shared_errors,) + tuple(curr_head.shape),
                                         dtype=curr_errors.dtype, device=curr_errors.device)
                                     * torch.ones((curr_errors_2.size(0) - shared_errors,) + tuple(
                                         curr_head_2.shape),
                                                  dtype=curr_errors.dtype, device=curr_errors.device)], dim=0)
            summand_errors = torch.cat([curr_errors_2[:shared_errors]
                                        * torch.ones_like(curr_errors[:shared_errors]),
                                        torch.zeros((curr_errors.size(0) - shared_errors,) + tuple(curr_head.shape),
                                                    dtype=curr_errors.dtype, device=curr_errors.device)
                                        * torch.ones(
                                            (curr_errors.size(0) - shared_errors,) + tuple(curr_head_2.shape),
                                            dtype=curr_errors.dtype, device=curr_errors.device),
                                        curr_errors_2[shared_errors:]
                                        * torch.ones(
                                            (curr_errors_2.size(0) - shared_errors,) + tuple(curr_head.shape),
                                            dtype=curr_errors.dtype, device=curr_errors.device)], dim=0)

            new_errors = self_errors + summand_errors

        new_errors = new_errors.transpose(1,0).view(-1, *new_head.shape[1:])
        curr_errors = curr_errors.transpose(1,0).view(-1, *curr_head.shape[1:])
        curr_errors_2 = curr_errors_2.transpose(1,0).view(-1, *curr_head_2.shape[1:])
        assert not torch.isnan(new_head).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return new_head, new_errors


class RelaxedAvgPool2d(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedAvgPool2d, self).__init__(layer)
        self.avg_pool2d = self.layer.avg_pool2d

    def forward(self, curr_head, curr_errors):
        curr_head, curr_errors = self.avg_pool2d(curr_head), self.avg_pool2d(curr_errors)
        return curr_head, curr_errors


class RelaxedGlobalAvgPool2d(RelaxedLayer):
    def __init__(self, layer):
        super(RelaxedGlobalAvgPool2d, self).__init__(layer)
        self.global_avg_pool2d = self.layer.global_avg_pool2d

    def forward(self, curr_head, curr_errors):
        curr_head, curr_errors = self.global_avg_pool2d(curr_head), self.global_avg_pool2d(curr_errors)
        return curr_head, curr_errors


class RelaxedFlatten(RelaxedLayer):
    def __init__(self,layer):
        super(RelaxedFlatten, self).__init__(layer)

    def forward(self, curr_head, curr_errors):
        curr_head = curr_head.view(curr_head.shape[0], -1)
        curr_errors = curr_errors.view(curr_errors.shape[0], -1)
        return curr_head, curr_errors


def bounds_from_cauchy(head, errors, n_rand_proj):
    l1_approx = 0
    errors = errors.view(head.shape[0], -1, *head.shape[1:])
    for i in range(0, errors.shape[1], n_rand_proj):
        l1_approx += torch.median(errors[:, i:i + n_rand_proj].abs(), dim=1)[0]
    errors = errors.view(-1, *head.shape[1:])
    return head - l1_approx, head + l1_approx