"""
Based on HybridZonotope from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import numpy as np
import torch
import torch.nn.functional as F


def clamp_image(x, eps, clamp_min=0, clamp_max=1):
    min_x = torch.clamp(x-eps, min=clamp_min)
    max_x = torch.clamp(x+eps, max=clamp_max)
    x_center = 0.5 * (max_x + min_x)
    x_beta = 0.5 * (max_x - min_x)
    return x_center, x_beta


def head_from_bounds(min_x,max_x):
    x_center = 0.5 * (max_x + min_x)
    x_betas = 0.5 * (max_x - min_x)
    return x_center, x_betas


def get_new_errs(should_box, newhead, newbeta):
    new_err_pos = (should_box.long().sum(dim=0) > 0).nonzero()
    num_new_errs = new_err_pos.size()[0]
    nnz = should_box.nonzero()
    if len(newhead.size()) == 2:
        batch_size, n = newhead.size()[0], newhead.size()[1]
        ids_mat = torch.zeros(n, dtype=torch.long).to(newhead.device)
        ids_mat[new_err_pos[:, 0]] = torch.arange(num_new_errs).to(newhead.device)
        beta_values = newbeta[nnz[:, 0], nnz[:, 1]]
        new_errs = torch.zeros((num_new_errs, batch_size, n)).to(newhead.device, dtype=newhead.dtype)
        err_ids = ids_mat[nnz[:, 1]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1]] = beta_values
    else:
        batch_size, n_channels, img_dim = newhead.size()[0], newhead.size()[1], newhead.size()[2]
        ids_mat = torch.zeros((n_channels, img_dim, img_dim), dtype=torch.long).to(newhead.device)
        ids_mat[new_err_pos[:, 0], new_err_pos[:, 1], new_err_pos[:, 2]] = torch.arange(num_new_errs).to(newhead.device)
        beta_values = newbeta[nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs = torch.zeros((num_new_errs, batch_size, n_channels, img_dim, img_dim)).to(newhead.device, dtype=newhead.dtype)
        err_ids = ids_mat[nnz[:, 1], nnz[:, 2], nnz[:, 3]]
        new_errs[err_ids, nnz[:, 0], nnz[:, 1], nnz[:, 2], nnz[:, 3]] = beta_values
    return new_errs


class HybridZonotope:

    def __init__(self, head, beta, errors, domain):
        self.head = head
        self.beta = beta
        self.errors = errors
        self.domain = domain
        self.device = self.head.device
        assert not torch.isnan(self.head).any()
        assert self.beta is None or (not torch.isnan(self.beta).any())
        assert self.errors is None or (not torch.isnan(self.errors).any())

    @staticmethod
    def construct_from_noise(x, eps, domain, dtype=torch.float32, data_range=(0,1)):
        # Clamp to data_range
        x_center, x_beta = clamp_image(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        if domain == 'box':
            return HybridZonotope(x_center, x_beta, None, domain)
        elif domain in ['zono', 'zono_iter', 'hbox']:
            batch_size = x.size()[0]
            n_elements = x[0].numel()
            ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(x.device)
            if len(x.size()) > 2:
                ei = ei.contiguous().view(n_elements, *x.size())

            new_beta = None if 'zono' in domain else torch.zeros(x_beta.shape).to(device=x_beta.device,
                                                                                  dtype=torch.float32)

            return HybridZonotope(x_center, new_beta, ei * x_beta.unsqueeze(0), domain)
        else:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))

    @staticmethod
    def construct_from_bounds(min_x, max_x, dtype=torch.float32, domain="box"):
        # Clamp to data_range
        x_center, x_beta = head_from_bounds(min_x,max_x)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        if domain == 'box':
            return HybridZonotope(x_center, x_beta, None, domain)
        elif domain in ['zono', 'zono_iter', 'hbox']:
            batch_size = x_center.size()[0]
            n_elements = x_center[0].numel()
            ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(x_center.device)
            if len(x_center.size()) > 2:
                ei = ei.contiguous().view(n_elements, *x_center.size())

            new_beta = None if "zono" in domain else torch.zeros(x_beta.shape).to(device=x_beta.device,
                                                                                  dtype=torch.float32)

            return HybridZonotope(x_center, new_beta, ei * x_beta.unsqueeze(0), domain)
        else:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))
    @staticmethod
    def join(x, trunk_errors=None, dim=0, mode="cat"):
        # x is list of HybridZonotopes
        # trunk_errors is number of last shared error between all Hybrid zonotopes, usually either number of initial
        # errors or number of errors at point where a split between network branches occured
        device = x[0].head.device
        if mode not in ["cat","stack"]:
            raise RuntimeError(f"Unkown join mode : {mode:}")

        if mode == "cat":
            new_head = torch.cat([x_i.head for x_i in x], dim=dim)
        elif mode == "stack":
            new_head = torch.stack([x_i.head for x_i in x], dim=dim)

        if all([x_i.beta is None for x_i in x]):
            new_beta = None
        elif any([x_i.beta is None for x_i in x]):
            assert False, "Mixed HybridZonotopes can't be joined"
        else:
            if mode == "cat":
                new_beta = torch.cat([x_i.beta for x_i in x], dim=dim)
            elif mode == "stack":
                new_beta = torch.stack([x_i.beta for x_i in x], dim=dim)

        if all([x_i.errors is None for x_i in x]):
            new_errors = None
        elif any([x_i.errors is None for x_i in x]):
            assert False, "Mixed HybridZonotopes can't be joined"
        else:
            if trunk_errors is None:
                trunk_errors = [0 for x_i in x]
            exit_errors = [x_i.errors.size()[0]-trunk_errors[i] for i, x_i in enumerate(x)] # number of additional errors for every Hybrid zonotope
            tmp_errors= [None for _ in x]
            for i, x_i in enumerate(x):
                tmp_errors[i] = torch.cat([x_i.errors[:trunk_errors[i]],
                                   torch.zeros([max(trunk_errors) - trunk_errors[i] + sum(exit_errors[:i])]
                                               + list(x_i.errors.size()[1:])).to(device),
                                   x_i.errors[trunk_errors[i]:],
                                   torch.zeros([sum(exit_errors[i + 1:])] + list(x_i.errors.size()[1:])).to(device)],
                                   dim=0)

            if mode == "cat":
                new_errors = torch.cat(tmp_errors, dim=dim + 1)
            elif mode == "stack":
                new_errors = torch.stack(tmp_errors, dim=dim+1)

        return HybridZonotope(new_head,
                              new_beta,
                              new_errors,
                              x[0].domain)

    def size(self, idx=None):
        if idx is None:
            return self.head.size()
        else:
            return self.head.size(idx)

    def view(self, size):
        return HybridZonotope(self.head.view(*size),
                              None if self.beta is None else self.beta.view(size),
                              None if self.errors is None else self.errors.view(self.errors.size()[0], *size),
                              self.domain)

    def normalize(self, mean, sigma):
        return (self - mean) / sigma

    def __sub__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head - other, self.beta, self.errors, self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __neg__(self):
        return self.neg()

    def __add__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head + other, self.beta, self.errors, self.domain)
        elif isinstance(other,HybridZonotope):
            assert self.domain == other.domain
            return self.add(other, shared_errors=0)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head / other,
                                  None if self.beta is None else self.beta / abs(other),
                                  None if self.errors is None else self.errors / other,
                                  self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __mul__(self, other):
        if isinstance(other, torch.Tensor):
            return HybridZonotope(self.head * other,
                                  None if self.beta is None else self.beta * abs(other),
                                  None if self.errors is None else self.errors * other,
                                  self.domain)
        else:
            assert False, 'Unknown type of other object'

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        return HybridZonotope(self.head[indices],
                              None if self.beta is None else self.beta[indices],
                              None if self.errors is None else self.errors[(slice(None), *indices)],
                              self.domain)

    def clone(self):
        return HybridZonotope(self.head.clone(),
                              None if self.beta is None else self.beta.clone(),
                              None if self.errors is None else self.errors.clone(),
                              self.domain)

    def detach(self):
        return HybridZonotope(self.head.detach(),
                              None if self.beta is None else self.beta.detach(),
                              None if self.errors is None else self.errors.detach(),
                              self.domain)

    def avg_pool2d(self, kernel_size, stride):
        new_head = F.avg_pool2d(self.head, kernel_size, stride)
        if self.beta is not None:
            new_beta = F.avg_pool2d(self.beta, kernel_size, stride)
        else:
            new_beta = None
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.head.shape[1:])
            new_errors = F.avg_pool2d(errors_resized, kernel_size, stride)
            new_errors = new_errors.view(-1, *new_head.shape)
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def global_avg_pool2d(self):
        new_head = F.adaptive_avg_pool2d(self.head, 1)
        if self.beta is not None:
            new_beta = F.adaptive_avg_pool2d(self.beta, 1)
        else:
            new_beta = None
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.head.shape[1:])
            new_errors = F.adaptive_avg_pool2d(errors_resized, 1)
            new_errors = new_errors.view(-1, *new_head.shape)
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def conv2d(self, weight, bias, stride, padding, dilation, groups):
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        new_beta = None if self.beta is None else F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv2d(errors_resized, weight, None, stride, padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def linear(self, weight, bias, C=None):
        if C is None:
            return self.matmul(weight.t()) + bias.unsqueeze(0)
        else:
            return self.unsqueeze(-1).rev_matmul(C.matmul(weight)).squeeze() + C.matmul(bias)

    def matmul(self, other):
        return HybridZonotope(self.head.matmul(other),
                              None if self.beta is None else self.beta.matmul(other.abs()),
                              None if self.errors is None else self.errors.matmul(other),
                              self.domain)
    def rev_matmul(self, other):
        return HybridZonotope(other.matmul(self.head),
                              None if self.beta is None else other.abs().matmul(self.beta),
                              None if self.errors is None else other.matmul(self.errors),
                              self.domain)

    def batch_norm_old(self, bn):
        view_dim_list = [1, -1]+(self.head.dim()-2)*[1]
        self_stat_dim_list = [0, 2, 3] if self.head.dim()==4 else [0, 2]
        if bn.training:
            momentum = 1 if bn.momentum is None else bn.momentum
            mean = self.head.mean(dim=self_stat_dim_list).detach()
            # var = (self.head - mean.view(*view_dim_list)).var(unbiased=False, dim=self_stat_dim_list).detach()
            var = self.head.var(unbiased=False, dim=self_stat_dim_list).detach()
            if bn.running_mean is not None and bn.running_var is not None and bn.track_running_stats:
                bn.running_mean = bn.running_mean * (1 - momentum) + mean * momentum
                bn.running_var = bn.running_var * (1 - momentum) + var * momentum
            else:
                bn.running_mean = mean
                bn.running_var = var
        c = (bn.weight / torch.sqrt(bn.running_var + bn.eps))
        b = (-bn.running_mean*c + bn.bias)
        new_head = self.head*c.view(*view_dim_list)+b.view(*view_dim_list)
        new_errors = None if self.errors is None else self.errors * c.view(*([1]+view_dim_list))
        new_beta = None if self.beta is None else self.beta * c.abs().view(*view_dim_list)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def batch_norm(self, bn, mean, var):
        view_dim_list = [1, -1]+(self.head.dim()-2)*[1]
        assert mean is not None and var is not None
        c = (bn.weight / torch.sqrt(var + bn.eps))
        b = (-mean*c + bn.bias)
        new_head = self.head*c.view(*view_dim_list)+b.view(*view_dim_list)
        new_errors = None if self.errors is None else self.errors * c.view(*([1]+view_dim_list))
        new_beta = None if self.beta is None else self.beta * c.abs().view(*view_dim_list)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def relu(self, deepz_lambda, bounds):
        lb, ub = self.concretize()
        D = 1e-6

        if self.domain == "box":
            min_relu, max_relu = F.relu(lb), F.relu(ub)
            return HybridZonotope(0.5 * (max_relu + min_relu), 0.5 * (max_relu - min_relu), None, self.domain), None
        elif self.domain == "hbox":
            is_under = (ub <= 0)
            is_above = (ub > 0) & (lb >= 0)
            is_cross = (ub > 0) & (lb < 0)

            new_head = self.head.clone()
            new_beta = self.beta.clone()
            new_errors = self.errors.clone()

            ub_half = ub / 2

            new_head[is_under] = 0
            new_head[is_cross] = ub_half[is_cross]

            new_beta[is_under] = 0
            new_beta[is_cross] = ub_half[is_cross]

            new_errors[:, ~is_above] = 0

            return HybridZonotope(new_head, new_beta, new_errors, self.domain), None
        elif "zono" in self.domain:
            if bounds is not None:
                lb_refined, ub_refined = bounds
                lb = torch.max(lb_refined, lb)
                ub = torch.min(ub_refined, ub)

            is_cross = (lb < 0) & (ub > 0)

            relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())
            if self.domain == 'zono_iter':
                if deepz_lambda is not None:
                    # assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()
                    if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                        deepz_lambda = relu_lambda
                    relu_lambda_cross = deepz_lambda
                else:
                    relu_lambda_cross = relu_lambda
                relu_mu_cross = torch.where(relu_lambda_cross < relu_lambda, 0.5*ub*(1-relu_lambda_cross), -0.5*relu_lambda_cross*lb)
                relu_lambda = torch.where(is_cross, relu_lambda_cross, (lb >= 0).float())
                relu_mu = torch.where(is_cross, relu_mu_cross, torch.zeros(lb.size()).to(self.device))
            else:
                relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size()).to(self.device))

            assert (not torch.isnan(relu_mu).any()) and (not torch.isnan(relu_lambda).any())

            new_head = self.head * relu_lambda + relu_mu
            old_errs = self.errors * relu_lambda
            new_errs = get_new_errs(is_cross, new_head, relu_mu)
            new_errors = torch.cat([old_errs, new_errs], dim=0)
            assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
            return HybridZonotope(new_head, None, new_errors, self.domain), deepz_lambda
        else:
            raise RuntimeError("Error applying ReLU with unkown domain: {}".format(self.domain))

    def log(self, deepz_lambda, bounds):
        lb, ub = self.concretize()
        D = 1e-6

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        assert (lb >= 0).all()

        if self.domain in ["box", "hbox"]:
            min_log, max_log = lb.log(), ub.log()
            return HybridZonotope(0.5 * (max_log + min_log), 0.5 * (max_log - min_log), None, "box"), None
        assert self.beta is None

        is_tight = (ub == lb)

        log_lambda_s = (ub.log() - (lb+D).log()) / (ub - lb + D)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((log_lambda_s - (1/ub)) / (1/(lb+D)-1/ub)).detach().requires_grad_(True)
            log_lambda = deepz_lambda * (1 / (lb+D)) + (1 - deepz_lambda) * (1 / ub)
        else:
            log_lambda = log_lambda_s

        log_lb_0 = torch.where(log_lambda < log_lambda_s, ub.log() - ub * log_lambda, lb.log() - lb * log_lambda)
        log_mu = 0.5 * (-log_lambda.log() - 1 + log_lb_0)
        log_delta = 0.5 * (-log_lambda.log() - 1 - log_lb_0)

        log_lambda = torch.where(is_tight, torch.zeros_like(log_lambda), log_lambda)
        log_mu = torch.where(is_tight, ub.log(), log_mu)
        log_delta = torch.where(is_tight, torch.zeros_like(log_delta), log_delta)

        assert (not torch.isnan(log_mu).any()) and (not torch.isnan(log_lambda).any())

        new_head = self.head * log_lambda + log_mu
        old_errs = self.errors * log_lambda
        new_errs = get_new_errs(~is_tight, new_head, log_delta)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain), deepz_lambda

    def exp(self, deepz_lambda, bounds):
        lb, ub = self.concretize()
        D = 1e-6

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        if self.domain in ["box", "hbox"]:
            min_exp, max_exp = lb.exp(), ub.exp()
            return HybridZonotope(0.5 * (max_exp + min_exp), 0.5 * (max_exp - min_exp), None, "box"), None

        assert self.beta is None

        is_tight = (ub.exp() - lb.exp()).abs() < 1e-15

        exp_lambda_s = (ub.exp() - lb.exp()) / (ub - lb + D)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((exp_lambda_s -lb.exp()) / (ub.exp()-lb.exp()+D)).detach().requires_grad_(True)
            exp_lambda = deepz_lambda * torch.min(ub.exp(), (lb + 1 - 0.05).exp()) + (1 - deepz_lambda) * lb.exp()
        else:
            exp_lambda = exp_lambda_s
        exp_lambda = torch.min(exp_lambda, (lb + 1 - 0.05).exp()) #torch.max(torch.min(exp_lambda, (lb + 1).exp()), torch.ones_like(lb)*D)  # ensure non-negative output only

        for _ in range(2):
            exp_ub_0 = torch.where(exp_lambda > exp_lambda_s, lb.exp() - exp_lambda * lb, ub.exp() - exp_lambda * ub)
            exp_mu = 0.5 * (exp_lambda * (1 - exp_lambda.log()) + exp_ub_0)
            exp_delta = 0.5 * (exp_ub_0 - exp_lambda * (1 - exp_lambda.log()))

            exp_lambda = torch.where(is_tight, torch.zeros_like(exp_lambda), exp_lambda)
            exp_mu = torch.where(is_tight, ub.exp(), exp_mu)
            exp_delta = torch.where(is_tight, torch.zeros_like(exp_delta), exp_delta)

            assert (not torch.isnan(exp_mu).any()) and (not torch.isnan(exp_lambda).any())

            new_head = self.head * exp_lambda + exp_mu
            old_errs = self.errors * exp_lambda
            new_errs = get_new_errs(~is_tight, new_head, exp_delta)
            new_errors = torch.cat([old_errs, new_errs], dim=0)
            assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
            new_zono = HybridZonotope(new_head, None, new_errors, self.domain)
            if (new_zono.concretize()[0] <= 0).any():
                exp_lambda = torch.where(new_zono.concretize()[0] <= 0, lb.exp(), exp_lambda) #torch.zeros_like(exp_lambda)
            else:
                break

        return HybridZonotope(new_head, None, new_errors, self.domain), deepz_lambda

    def inv(self, deepz_lambda, bounds):
        lb, ub = self.concretize()
        assert (lb > 0).all()

        if self.domain in ["box", "hbox"]:
            min_inv, max_inv = 1 / ub, 1 / lb
            return HybridZonotope(0.5 * (max_inv + min_inv), 0.5 * (max_inv - min_inv), None, "box"), None

        assert self.beta is None

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        assert (lb > 0).all()

        is_tight = (ub == lb)

        inv_lambda_s = -1 / (ub * lb)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            # assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = (-ub*lb+lb**2)/(lb**2-ub**2)
            inv_lambda = deepz_lambda * (-1 / lb**2) + (1 - deepz_lambda) * (-1 / ub**2)
        else:
            inv_lambda = inv_lambda_s


        inv_ub_0 = torch.where(inv_lambda > inv_lambda_s, 1 / lb - inv_lambda * lb, 1 / ub - inv_lambda * ub)
        inv_mu = 0.5 * (2 * (-inv_lambda).sqrt() + inv_ub_0)
        inv_delta = 0.5 * (inv_ub_0 - 2 * (-inv_lambda).sqrt())

        # inv_mu = torch.where(inv_lambda > inv_lambda_s, 0.5 * (2 * (-inv_lambda).sqrt() + 1 / lb - inv_lambda * lb),
        #                      0.5 * (2 * (-inv_lambda).sqrt() + 1 / ub - inv_lambda * ub))

        inv_lambda = torch.where(is_tight, torch.zeros_like(inv_lambda), inv_lambda)
        inv_mu = torch.where(is_tight, 1/ub, inv_mu)
        inv_delta = torch.where(is_tight, torch.zeros_like(inv_delta), inv_delta)

        assert (not torch.isnan(inv_mu).any()) and (not torch.isnan(inv_lambda).any())

        new_head = self.head * inv_lambda + inv_mu
        old_errs = self.errors * inv_lambda
        new_errs = get_new_errs(~is_tight, new_head, inv_delta)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain), deepz_lambda

    def sum(self, dim, reduce_dim=False):
        new_head = self.head.sum(dim=dim)
        new_beta = None if self.beta is None else self.beta.abs().sum(dim=dim)
        new_errors = None if self.errors is None else self.errors.sum(dim=dim+1)

        if not reduce_dim:
            new_head = new_head.unsqueeze(dim)
            new_beta = None if new_beta is None else new_beta.unsqueeze(dim)
            new_errors = None if new_errors is None else new_errors.unsqueeze(dim+1)

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def unsqueeze(self, dim):
        new_head = self.head.unsqueeze(dim)
        new_beta = None if self.beta is None else self.beta.unsqueeze(dim)
        new_errors = None if self.errors is None else self.errors.unsqueeze(dim+1)

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def squeeze(self, dim=None):
        if dim is None:
            new_head = self.head.squeeze()
            new_beta = None if self.beta is None else self.beta.squeeze()
            new_errors = None if self.errors is None else self.errors.squeeze()
        else:
            new_head = self.head.squeeze(dim)
            new_beta = None if self.beta is None else self.beta.squeeze(dim)
            new_errors = None if self.errors is None else self.errors.squeeze(dim+1)

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def neg(self):
        new_head = -self.head
        new_beta = None if self.beta is None else self.beta
        new_errors = None if self.errors is None else -self.errors
        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def add(self, summand_zono, shared_errors=0):
        assert all([x == y or x == 1 or y == 1 for x, y in zip(self.head.shape[::-1], summand_zono.head.shape[::-1])])
        #assert self.domain == summand_zono.domain

        new_head = self.head + summand_zono.head
        if self.beta is None and summand_zono.beta is None:
            new_beta = None
        elif self.beta is not None and summand_zono.beta is not None:
            new_beta = self.beta.abs() + summand_zono.beta.abs()
        else:
            new_beta = self.beta if self.beta is not None else summand_zono.beta

        if self.errors is None:
            new_errors = None
        elif self.errors is not None and summand_zono.errors is not None:
            if shared_errors < 0:
                shared_errors = self.errors.size(0) if self.errors.size(0) == summand_zono.errors.size(0) else 0

            #Shape cast errors to output shape
            self_errors = torch.cat([self.errors
                                     * torch.ones((self.errors.size(0),) + tuple(summand_zono.head.shape),
                                                  dtype=self.errors.dtype, device=self.errors.device),
                                     torch.zeros((summand_zono.errors.size(0) - shared_errors,)+ tuple(self.head.shape),
                                                   dtype=self.errors.dtype, device=self.errors.device)
                                     * torch.ones((summand_zono.errors.size(0) - shared_errors,) + tuple(summand_zono.head.shape),
                                                   dtype=self.errors.dtype, device=self.errors.device)], dim=0)
            summand_errors = torch.cat([summand_zono.errors[:shared_errors]
                                       * torch.ones_like(self.errors[:shared_errors]),
                                       torch.zeros((self.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                     dtype=self.errors.dtype, device=self.errors.device)
                                       * torch.ones((self.errors.size(0) - shared_errors,) + tuple(summand_zono.head.shape),
                                                     dtype=self.errors.dtype, device=self.errors.device),
                                       summand_zono.errors[shared_errors:]
                                       * torch.ones((summand_zono.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                     dtype=self.errors.dtype, device=self.errors.device)], dim=0)

            new_errors = self_errors + summand_errors
        else:
            new_errors = self.errors if self.errors is not None else summand_zono.errors


        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        new_domain = summand_zono.domain if new_beta is None else ("hbox" if new_errors is not None else "box")
        return HybridZonotope(new_head, new_beta, new_errors, new_domain)

    def prod(self, factor_zono, shared_errors=None, low_mem=False):
        lb_self, ub_self = self.concretize()
        lb_other, ub_other = factor_zono.concretize()
        if self.domain == factor_zono.domain:
            domain = self.domain
        elif "box" in [self.domain, factor_zono.domain]:
            domain = "box"
        elif "hbox" in [self.domain, factor_zono.domain]:
            domain = "hbox"
        else:
            assert False

        if domain in ["box", "hbox"] or low_mem:
            min_prod = torch.min(torch.min(torch.min(lb_self*lb_other, lb_self*ub_other), ub_self*lb_other), ub_self*ub_other)
            max_prod = torch.max(torch.max(torch.max(lb_self*lb_other, lb_self*ub_other), ub_self*lb_other), ub_self*ub_other)
            return HybridZonotope(0.5 * (max_prod + min_prod), 0.5 * (max_prod - min_prod), None, "box")
        assert self.beta is None

        assert all([x==y or x==1 or y ==1 for x,y in zip(self.head.shape[::-1],factor_zono.head.shape[::-1])])
        if shared_errors is None:
            shared_errors = 0
        if shared_errors == -1:
            shared_errors = self.errors.size(0) if self.errors.size(0) == factor_zono.errors.size(0) else 0

        #Shape cast to output shape
        self_errors = torch.cat([self.errors
                                 * torch.ones((self.errors.size(0),) + tuple(factor_zono.head.shape),
                                              dtype=self.errors.dtype, device=self.errors.device),
                                 torch.zeros((factor_zono.errors.size(0) - shared_errors,)+ tuple(self.head.shape),
                                               dtype=self.errors.dtype, device=self.errors.device)
                                 * torch.ones((factor_zono.errors.size(0) - shared_errors,) + tuple(factor_zono.head.shape),
                                               dtype=self.errors.dtype, device=self.errors.device)], dim=0)
        factor_errors = torch.cat([factor_zono.errors[:shared_errors]
                                   * torch.ones_like(self.errors[:shared_errors]),
                                   torch.zeros((self.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                 dtype=self.errors.dtype, device=self.errors.device)
                                   * torch.ones((self.errors.size(0) - shared_errors,) + tuple(factor_zono.head.shape),
                                                 dtype=self.errors.dtype, device=self.errors.device),
                                   factor_zono.errors[shared_errors:]
                                   * torch.ones((factor_zono.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                 dtype=self.errors.dtype, device=self.errors.device)], dim=0)

        lin_err = self.head.unsqueeze(dim=0) * factor_errors + factor_zono.head.unsqueeze(dim=0) * self_errors
        quadr_const = (self_errors * factor_errors)
        quadr_error_tmp = self_errors.unsqueeze(1) * factor_errors.unsqueeze(0)
        quadr_error_tmp = 1./2. * (quadr_error_tmp + quadr_error_tmp.transpose(1, 0)).abs().sum(dim=1).sum(dim=0)\
                            - 1./2. * quadr_const.abs().sum(dim=0)

        new_head = self.head * factor_zono.head + 1. / 2. * quadr_const.sum(dim=0)
        old_errs = lin_err
        new_errs = get_new_errs(torch.ones(self.head.shape), new_head, quadr_error_tmp)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain)

    def upsample(self, size, mode, align_corners, consolidate_errors=True):
        new_head = F.interpolate(self.head, size=size, mode=mode, align_corners=align_corners)
        delta = 0

        if self.beta is not None:
            new_beta = F.interpolate(self.beta, size=size, mode=mode, align_corners=align_corners)
            delta = delta + new_beta
        else:
            new_beta = None

        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.head.shape[1:])
            new_errors = F.interpolate(errors_resized, size=size, mode=mode, align_corners=align_corners)
            new_errors = new_errors.view(-1, *new_head.shape)
            delta = delta + new_errors.abs().sum(0)
        else:
            new_errors = None

        if consolidate_errors:
            return HybridZonotope.construct_from_bounds(new_head-delta, new_head+delta, domain=self.domain)
        else:
            return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    def concretize(self):
        delta = 0
        if self.beta is not None:
            delta = delta + self.beta
        if self.errors is not None:
            delta = delta + self.errors.abs().sum(0)
        return self.head - delta, self.head + delta

    def avg_width(self):
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(self, i, j, threshold_min=0):
        diff_head = self.head[:, i] - self.head[:, j]
        delta = diff_head
        if self.errors is not None:
            diff_errors = (self.errors[:, :, i] - self.errors[:, :, j]).abs().sum(dim=0)
            delta -= diff_errors
        if self.beta is not None:
            diff_beta = (self.beta[:, i] + self.beta[:, j]).abs()
            delta -= diff_beta
        return delta, delta > threshold_min

    def verify(self, targets, target_list=None, threshold_min=0, corr_only=False):
        n_class = self.head.size()[1]
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        if n_class == 1:
            # assert len(targets) == 1
            verified_list = torch.cat([self.concretize()[1] < threshold_min,self.concretize()[0] >= threshold_min], dim=1) #[self.concretize()[1]<0,self.concretize()[0]>=0]
            verified[:] = torch.any(verified_list, dim=1)
            verified_corr[:] = verified_list.gather(dim=1,index=targets.long().unsqueeze(dim=1)).squeeze(1)#verified_list[targets]
            threshold = torch.cat(self.concretize(),1).gather(dim=1, index=(1-targets).long().unsqueeze(dim=1)).squeeze(1)#self.concretize()[1-targets]
        elif target_list is not None:
            assert len(targets) == 1
            threshold = 5000 * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
            verified_corr[0] = 1
            for i in range(n_class):
                # => for every non target class there is a target class that is always grater => ver
                if not verified_corr:
                    break
                if (i not in target_list) == targets:
                    # i will be a non correct class
                    isg = torch.zeros_like(verified)
                    margin = -5000 * torch.ones_like(threshold)
                    for j in range(n_class):
                        if (j in target_list) == targets:
                            # j will be a correct class
                            margin_tmp, ok = self.is_greater(j, i, threshold_min)
                            margin = max(margin, margin_tmp)
                            isg = isg | ok.byte()
                    threshold = min(threshold, margin)
                    verified_corr = verified_corr & isg
            verified[0] = verified_corr
        else:
            threshold = 5000 * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
            for i in range(n_class):
                if corr_only and i not in targets:
                    continue
                isg = torch.ones(targets.size(), dtype=torch.uint8).to(self.head.device)
                margin = 5000 * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
                for j in range(n_class):
                    if i != j and isg.any():
                        margin_tmp, ok = self.is_greater(i, j, threshold_min)
                        margin = torch.min(margin, margin_tmp)
                        isg = isg & ok.byte()
                verified = verified | isg
                verified_corr = verified_corr | (targets.eq(i).byte() & isg)
                threshold = torch.where(targets.eq(i).byte(), margin, threshold)
        return verified, verified_corr, threshold

    def get_min_diff(self, i, j):
        """ returns minimum of logit[i] - logit[j] """
        return self.is_greater(i, j)[0]

    def get_wc_logits(self, targets, deltas=False):
        n_class = self.size(-1)
        device = self.head.device

        if deltas:
            def get_c_mat(n_class, target):
                return torch.eye(n_class, dtype=torch.float32)[target].unsqueeze(dim=0) \
                       - torch.eye(n_class, dtype=torch.float32)
            if n_class > 1:
                c = torch.stack([get_c_mat(n_class,x) for x in targets], dim=0)
                self = -(self.unsqueeze(dim=1) * c.to(device)).sum(dim=2, reduce_dim=True)
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        if n_class == 1:
            wc_logits = torch.cat([ub, lb],dim=1)
            wc_logits = wc_logits.gather(dim=1, index=targets.long().unsqueeze(1))
        else:
            wc_logits = ub.clone()
            wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits

    def ce_loss(self, targets):
        wc_logits = self.get_wc_logits(targets)
        if wc_logits.size(1) == 1:
            return F.binary_cross_entropy_with_logits(wc_logits.squeeze(1), targets.float(), reduction="none")
        else:
            return F.cross_entropy(wc_logits, targets.long(), reduction="none")
