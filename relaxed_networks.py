import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (
    Conv2d, Normalization, Flatten, Linear, ReLU, BatchNorm2d, BatchNorm1d,
    AvgPool2d, Bias, Scale, Sequential, Entropy, GlobalAvgPool2d, Upsample)
from relaxed_layers import (RelaxedConv2d, RelaxedNorm, RelaxedBatchNorm2d, RelaxedBatchNorm1d, RelaxedReLU, RelaxedAvgPool2d, RelaxedFlatten, RelaxedLinear,
    RelaxedBias, RelaxedScale, RelaxedEntropy, RelaxedGlobalAvgPool2d, RelaxedUpsample)
from zonotope import clamp_image, HybridZonotope
from utils import my_cauchy
import sys
sys.path.append("../auto_LiRPA_Repo")
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm, BoundDataParallel, CrossEntropyWrapper
from auto_LiRPA.bound_ops import BoundExp, BoundSub

import numpy as np

def get_adv_output(cnet, net, relaxed_net, inputs, targets, adv_attack, layer_idx, is_train):
    device = inputs.device
    eps, n_steps, step_size, adv_type = adv_attack.get_params()
    adv_errors = {}
    net.eval()
    if relaxed_net is not None:
        relaxed_net.eval()

    with torch.enable_grad():
        for it in range(n_steps + 1):
            curr_head, A_0 = clamp_image(inputs, eps)
            if it == 0:
                adv_errors[-1] = torch.FloatTensor(curr_head.shape).unsqueeze(1).to(device).uniform_(-1, 1).requires_grad_(True)
            curr_errors = A_0.unsqueeze(1) * adv_errors[-1]

            if layer_idx >= 0:
                adv_latent, res_latent = relaxed_net.forward(curr_head, curr_errors, end_layer=layer_idx, adv_errors=adv_errors)
            else:
                adv_latent = curr_head + curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:]).sum(dim=1)
                res_latent = None

            if it == n_steps:
                break

            adv_outs = net.forward(adv_latent, res_latent, layer_idx)

            if adv_type == "pgd":
                ce_loss = cnet.lossFn(adv_outs, targets).sum(dim=0)
            elif adv_type == "Trades":
                ce_loss = nn.KLDivLoss(reduction='sum')(F.log_softmax(adv_outs, dim=1),
                                                        F.softmax(net(inputs), dim=1)).mean(dim=-1)
            cnet.zero_grad()
            ce_loss.backward()
            for i in adv_errors:
                adv_errors[i].data = torch.clamp(adv_errors[i].data + step_size * adv_errors[i].grad.sign(), -1, 1)
                adv_errors[i].grad.zero_()
        if is_train:
            net.train()
            if relaxed_net is not None:
                relaxed_net.train()

        adv_latent = adv_latent.detach()
        if res_latent is not None:
            res_latent = res_latent.detach()
        adv_outs = net.forward(adv_latent, res_latent, layer_idx)
    return adv_outs


def get_bound_derivates(net, relaxed_net, inputs, adv_attack, layer_ids, stable_idx, stable_losses,
                        relu_stable_type, is_train):
    eps, n_steps, step_size, adv_type = adv_attack.get_params()
    relaxed_net.eval()

    n_rand_proj = relaxed_net.n_rand_proj

    curr_head, A_0 = clamp_image(inputs, eps)
    curr_cauchy = A_0.unsqueeze(1) * my_cauchy(inputs.shape[0], n_rand_proj, *inputs.shape[1:]).to(
        curr_head.device)

    assert not torch.isinf(curr_cauchy).any()

    end_layer = max(layer_ids) if stable_idx is None else max(max(layer_ids), max(stable_idx))
    bounds = relaxed_net.forward(curr_head, curr_cauchy, end_layer=end_layer, compute_bounds=True)

    for layer_idx, (lb, ub) in bounds.items():
        if layer_idx == len(net.blocks):
            continue
        if relu_stable_type == 'tight':
            stable_losses[layer_idx] = (torch.clamp(-lb, min=0) * torch.clamp(ub, min=0)).mean()
        elif relu_stable_type == 'tanh':
            stable_losses[layer_idx] = -torch.tanh(1 + lb * ub).mean()
        elif relu_stable_type == 'mixed':
            stable_losses[layer_idx] = (-torch.tanh(1 + lb * ub * 0.01) - torch.clamp(lb * ub, max=0) * 0.001).mean()
        else:
            assert False, 'Unknown type of ReLU regularization!'
        assert not torch.isnan(stable_losses[layer_idx]).any()
    if is_train:
        relaxed_net.train()
    return bounds, stable_losses


class CombinedNetwork(nn.Module):
    def __init__(self, net, relaxed_net, lossFn=nn.CrossEntropyLoss(reduction='none'),
                 evalFn=lambda x: torch.max(x,dim=1)[1], evalFn_test=None, lossFn_test=None, targetFn_test=None,
                 threshold_min=0, bound_opts=None, dummy_input=None, device=None, no_r_net=False):
        super(CombinedNetwork, self).__init__()
        self.net = net
        self.relaxed_net = relaxed_net
        self.lossFn = lossFn
        self.evalFn = evalFn
        self.lossFn_test = lossFn_test
        self.evalFn_test = evalFn_test
        self.targetFn_test = targetFn_test
        self.threshold_min = threshold_min

        if net is None or no_r_net:
            self.lirpa_net = None
            self.lirpa_net_loss = None
        else:
            # device = device if device is not None else net.device
            self.lirpa_net = BoundedModule(net, dummy_input, bound_opts={'relu': bound_opts}, device=device)
            self.lirpa_net_loss = BoundedModule(CrossEntropyWrapper(net), (dummy_input, torch.zeros(1, dtype=torch.long)),
                                       bound_opts={'relu': bound_opts, 'loss_fusion': True}, device=device)
            self.lirpa_net_loss = BoundDataParallel(self.lirpa_net_loss)


    def update_loss_fn(self, balance_factor, device, testFn = False):
        balance_factor = torch.tensor(balance_factor, dtype=torch.float, device=device)
        def lossFn_updated(x, y): return torch.nn.functional.binary_cross_entropy_with_logits(x.view(-1), y.float(), pos_weight=balance_factor, reduction='none')
        if testFn:
            if hasattr(self, "lossFn_test"):
                setattr(self, "lossFn_test", lossFn_updated)
        else:
            if hasattr(self, "lossFn"):
                setattr(self, "lossFn", lossFn_updated)

    @staticmethod
    def get_cnet(net, device, lossFn, evalFn, n_rand_proj, threshold_min=0, no_r_net=False, lirpa_bound_opts=None, input_channels=None):
        if no_r_net:
            relaxed_net = None
        else:
            relaxed_net = RelaxedNetwork(net.blocks, n_rand_proj).to(device)

        if net is None:
            dummy_input = None
        else:
            if net.dims is None:
                assert input_channels is not None
                input_size = (input_channels, net.net_dim, net.net_dim)
            else:
                input_size = net.dims[0]

            dummy_input = torch.rand((1,)+input_size, device=device, dtype=torch.float32)

        cnet = CombinedNetwork(net, relaxed_net, lossFn=lossFn, evalFn=evalFn, threshold_min=threshold_min,
                               bound_opts=lirpa_bound_opts, dummy_input=dummy_input, device=device, no_r_net=no_r_net).to(device)
        return cnet

    def get_exp_module(self):
        for _, node in self.lirpa_net_loss.named_modules():
            # Find the Exp neuron in computational graph
            if isinstance(node, BoundExp):
                return node
        return None

    def get_LiRPA_losses(self, inputs, targets_abs, eps, domain, threshold_min=0, beta=0.5):
        data_ub = torch.clamp(inputs + eps, max=1)
        data_lb = torch.clamp(inputs - eps, min=0)
        n_class = self.net.n_class
        final_node_name = None
        loss_fusion = domain.startswith("lf")
        domain = "l" + domain[2:] if domain.startswith("lf") else domain
        targets_abs = targets_abs.to(torch.long)

        model = self.lirpa_net_loss if loss_fusion else self.lirpa_net

        ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(inputs, ptb)
        model(x, targets_abs)
        x = (x, targets_abs)
        I = (~(targets_abs.data.unsqueeze(1) == torch.arange(n_class).type_as(targets_abs.data).unsqueeze(0)))
        bound_lower, bound_upper = False, False

        if loss_fusion:
            bound_upper = True
            c = None
        else:
            if n_class > 1:
                c = torch.eye(n_class).type_as(inputs)[targets_abs].unsqueeze(1) - torch.eye(n_class).type_as(inputs).unsqueeze(
                    0)
                # remove specifications to self
                c = (c[I].view(inputs.size(0), n_class - 1, n_class))
                bound_lower = True
            else:
                c = None

        if n_class == 1:
                bound_lower = True
                bound_upper = True


        if domain == 'l_IBP':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                           final_node_name=final_node_name, no_replicas=True)
        elif domain == 'l_CROWN':
            lb, ub = model(method_opt="compute_bounds", x=x, IBP=False, C=c, method='backward',
                           bound_lower=bound_lower, bound_upper=bound_upper)
        elif domain == 'l_CROWN-IBP':
            # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method='backward')  # pure IBP bound
            # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
            ilb, iub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                             final_node_name=final_node_name, no_replicas=True)
            clb, cub = model(method_opt="compute_bounds", IBP=False, C=c, method='backward',
                             bound_lower=bound_lower, bound_upper=bound_upper, final_node_name=final_node_name,
                             no_replicas=True)

            if bound_upper:
                ub = cub * beta + iub * (1 - beta)
            if bound_lower:
                lb = clb * beta + ilb * (1 - beta)

        if bound_upper:
            ub_margin = ub if not domain == 'l_CROWN-IBP' else torch.min(iub, cub)
        if bound_lower:
            lb_margin = lb if not domain == 'l_CROWN-IBP' else torch.max(ilb, clb)

        if n_class>1:
            if loss_fusion:
                node = getattr(model.module, model.module.output_name[0])
                while not isinstance(node, BoundSub):
                    node = getattr(model.module,node.input_name[0])
                margin = torch.min(node.interval[0][I].view(targets_abs.size(0),-1),dim=-1)[0]
            else:
                margin = torch.min(lb_margin.view(targets_abs.size(0),-1),dim=-1)[0]
            threshold_min_temp = threshold_min
        else:
            margin = torch.stack((-ub_margin,lb_margin),dim=0).gather(dim=0, index=targets_abs.view(1, -1, 1)).view(-1)

            threshold_min_temp = (-1 + 2 * targets_abs) * threshold_min
        ver_ok = (margin > threshold_min_temp).to(torch.float)

        if loss_fusion:
            if isinstance(model, BoundDataParallel):
                max_input = model(get_property=True, node_class=BoundExp, att_name='max_input')
            else:
                max_input = self.get_exp_module().max_input
            return ver_ok, margin, torch.mean(torch.log(ub) + max_input)
        else:
            # Pad zero at the beginning for each example, and use fake label '0' for all examples
            if n_class>1:
                lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
                fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
                robust_loss = self.lossFn(-lb_padded, fake_labels)
            else:
                loss_margin = torch.stack((ub_margin,lb_margin),dim=0).gather(dim=0, index=targets_abs.view(1, -1, 1)).view(-1)
                robust_loss = self.lossFn(loss_margin, targets_abs)
            return ver_ok, margin, robust_loss

    def get_abs_loss(self, inputs, targets, eps, domain, threshold_min, beta=None):
        if domain in ["zono", "box", 'zono_iter', 'hbox']:
            _, ver_corr, threshold_n, abs_outputs = self.net.verify(inputs, targets, eps, domain, threshold_min, return_abs=True)

            if domain in ["box","hbox"] and self.net.n_class > 1:
                abs_loss = self.lossFn(-abs_outputs.concretize()[0], targets)
            else:
                abs_loss = self.lossFn(abs_outputs.get_wc_logits(targets, True), targets)  # abs_outputs.ce_loss(targets_abs)
        elif domain in ["l_IBP", "l_CROWN", "l_CROWN-IBP", "lf_IBP", "lf_CROWN", "lf_CROWN-IBP"]:
            ver_corr, threshold_n, abs_loss = self.get_LiRPA_losses(inputs, targets, eps, domain, self.threshold_min, beta=beta)
        else:
            raise RuntimeError(f"domain {domain} is not known")

        return ver_corr.to(torch.float), threshold_n, abs_loss

    def forward(self, inputs, targets, args, stable_idx=None, adv_attack=None, layer_ids=[-1],
                   is_train=False, relu_stable_type="tight", robust_loss_mode=None, train_mode=None, domains=None,beta=1):
        net = self.net
        relaxed_net = self.relaxed_net
        device = inputs.device
        train_mode = args.train_mode if train_mode is None else train_mode
        domains = [] if domains is None else (domains if isinstance(domains, list) else [domains])

        adv_loss, adv_ok, adv_y = {}, {}, {}
        stable_losses = {}
        abs_loss, abs_ok = {}, {}

        targets_nat = targets if not isinstance(targets, dict) else targets[-2]
        nat_outs = net.forward(inputs).view(targets_nat.size(0), -1)
        if is_train or self.lossFn_test is None or self.evalFn_test is None:
            nat_loss = self.lossFn(nat_outs, targets_nat)
            nat_y = self.evalFn(nat_outs)
            nat_ok = targets_nat.eq(nat_y).float()
        else:
            nat_loss = self.lossFn_test(nat_outs, targets_nat)
            nat_y = self.evalFn_test(nat_outs)
            nat_ok = self.targetFn_test(targets_nat).eq(nat_y).float()
        adv_loss[-2] = nat_loss
        adv_ok[-2] = nat_ok
        threshold = None

        if (len(domains) == 0 or domains[0] is None) and max(layer_ids) == -2:
            return nat_loss, nat_ok, nat_y, adv_loss, adv_ok, adv_y, abs_loss, abs_ok, stable_losses, threshold

        if adv_attack is not None:
            eps, n_steps, step_size, adv_type = adv_attack.get_params()
        n_class = nat_outs.size(1)

        if "diffAI" in train_mode:
            for domain in domains:
                targets_abs = targets if not isinstance(targets, dict) else targets[domain]
                abs_ok[domain], _, abs_loss[domain] = self.get_abs_loss(inputs, targets_abs, eps, domain, self.threshold_min,beta=beta)

        if max(layer_ids) == -2:
            # Only natural loss required
            return nat_loss, nat_ok, nat_y, adv_loss, adv_ok, adv_y, abs_loss, abs_ok, stable_losses, threshold

        if stable_idx is not None or max(layer_ids)>-1:
            assert relaxed_net is not None
            bounds, stable_losses = get_bound_derivates(net, relaxed_net, inputs, adv_attack, layer_ids,
                                               stable_idx, stable_losses, relu_stable_type, is_train)

            targets_stable = targets if not isinstance(targets, dict) else targets[-2]
            bounds, stable_losses = get_bound_derivates(net, relaxed_net, inputs, adv_attack, layer_ids,
                                               stable_idx, stable_losses, relu_stable_type, is_train)
        for layer_idx in layer_ids:
            if layer_idx < -1:
                continue
            if layer_idx >= 0:
                assert relaxed_net is not None

            targets_adv = targets if not isinstance(targets, dict) else targets[layer_idx]
            adv_outs = get_adv_output(self, net, relaxed_net, inputs, targets_adv, adv_attack, layer_idx, is_train)

            if adv_type == "pgd":
                adv_loss[layer_idx] = self.lossFn(adv_outs, targets_adv)
            elif adv_type == "Trades":
                adv_loss[layer_idx] = nn.KLDivLoss(reduction='none')(F.log_softmax(adv_outs, dim=1),
                                                                     F.softmax(net(inputs), dim=1)).mean(dim=-1)

            if is_train or self.lossFn_test is None or self.evalFn_test is None:
                adv_y[layer_idx] = self.evalFn(adv_outs)
                adv_ok[layer_idx] = targets_adv.eq(adv_y[layer_idx]).float()
            else:
                adv_y[layer_idx] = self.evalFn_test(adv_outs)
                adv_ok[layer_idx] = self.targetFn_test(targets_adv).eq(adv_y[layer_idx]).float()

            if layer_idx == -1:
                if adv_outs.size(1) == 1:
                    threshold = adv_outs
                else:
                    threshold = adv_outs.max(dim=1)[0] - \
                                adv_outs[torch.arange(n_class).cuda().unsqueeze(dim=0) != targets_adv.unsqueeze(dim=1)].view(-1, n_class-1).max(dim=1)[0]

        return nat_loss, nat_ok, nat_y, adv_loss, adv_ok, adv_y, abs_loss, abs_ok, stable_losses, threshold


class RelaxedNetwork(nn.Module):

    def __init__(self, blocks=None, n_rand_proj=None):
        super(RelaxedNetwork, self).__init__()
        self.blocks = blocks
        if blocks is None:
            self.relaxed_blocks = None
        else:
            relaxed_blocks = RelaxedNetwork.get_relaxed_blocks(blocks, n_rand_proj)
            self.relaxed_blocks = Sequential(*relaxed_blocks)
        self.n_rand_proj = n_rand_proj

    def get_relu_ids(self):
        relu_ids = []
        for idx, relaxed_block in enumerate(self.relaxed_blocks):
            if isinstance(relaxed_block, RelaxedReLU):
                relu_ids += [idx]
        return relu_ids

    def freeze(self, layer_idx):
        for i in range(layer_idx+1):
            self.relaxed_blocks[i].requires_grad_(False)

    @staticmethod
    def get_relaxed_blocks(blocks, n_rand_proj):
        relaxed_blocks = []
        for block in blocks:
            if isinstance(block, Conv2d):
                relaxed_blocks += [RelaxedConv2d(block)]
            elif isinstance(block, Normalization):
                relaxed_blocks += [RelaxedNorm(block)]
            elif isinstance(block, BatchNorm2d):
                relaxed_blocks += [RelaxedBatchNorm2d(block)]
            elif isinstance(block, BatchNorm1d):
                relaxed_blocks += [RelaxedBatchNorm1d(block)]
            elif isinstance(block, ReLU):
                relaxed_blocks += [RelaxedReLU(block)]
            elif isinstance(block, AvgPool2d):
                relaxed_blocks += [RelaxedAvgPool2d(block)]
            elif isinstance(block, GlobalAvgPool2d):
                relaxed_blocks += [RelaxedGlobalAvgPool2d(block)]
            elif isinstance(block, Flatten):
                relaxed_blocks += [RelaxedFlatten(block)]
            elif isinstance(block, Linear):
                relaxed_blocks += [RelaxedLinear(block)]
            elif isinstance(block, Bias):
                relaxed_blocks += [RelaxedBias(block)]
            elif isinstance(block, Scale):
                relaxed_blocks += [RelaxedScale(block)]
            elif isinstance(block, Entropy):
                relaxed_blocks += [RelaxedEntropy(block)]
            elif isinstance(block, Upsample):
                relaxed_blocks += [RelaxedUpsample(block)]
            elif isinstance(block, Sequential):
                try:
                    relaxed_blocks += RelaxedNetwork.get_relaxed_blocks([*block.layers._modules['0'].residual.layers], n_rand_proj)
                except:
                    assert False, "Sequential block mishandled"
            else:
                assert False, 'Unknown layer type: {}'.format(type(block))
        return relaxed_blocks

    @staticmethod
    def propagate(relaxed_blocks, curr_head, curr_errors, is_cauchy, n_rand_proj, layer_idx=None, adv_errors=None, bounds =None, adv_error_pre=None):
        curr_errors = curr_errors.view(-1, *curr_head.shape[1:])
        res_head, res_errors = None, None
        bounds = {} if bounds is None else bounds
        for j, relaxed_layer in enumerate(relaxed_blocks):
            if j in bounds:
                relaxed_layer.bounds = (bounds[j][0].detach(), bounds[j][1].detach())
            if layer_idx is not None and j > layer_idx:
                continue
            if isinstance(relaxed_layer, RelaxedReLU):
                if is_cauchy:
                    curr_head, curr_errors = relaxed_layer(curr_head, curr_errors, k=n_rand_proj)
                else:
                    adv_error_id = "%d" % j if adv_error_pre is None else "%s_%d" %(adv_error_pre,j)
                    if adv_error_id not in adv_errors:
                        adv_errors[adv_error_id] = torch.FloatTensor(curr_head.shape).unsqueeze(1).to(curr_head.device)\
                                            .uniform_(-1, 1).requires_grad_(True)
                    curr_head, curr_errors = relaxed_layer(curr_head, curr_errors, adv_errors=adv_errors[adv_error_id])
            elif isinstance(relaxed_layer, RelaxedEntropy):
                curr_head, curr_errors = relaxed_layer(curr_head, curr_errors, is_cauchy, adv_errors, n_rand_proj, adv_error_pre)
            else:
                curr_head, curr_errors = relaxed_layer(curr_head, curr_errors)
            if is_cauchy and j < len(relaxed_blocks):
                l1_approx = 0
                curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:])
                for i in range(0, curr_errors.shape[1], n_rand_proj):
                    l1_approx += torch.median(curr_errors[:, i:i+n_rand_proj].abs(), dim=1)[0]
                curr_errors = curr_errors.view(-1, *curr_head.shape[1:])
                bounds[j+1] = (curr_head - l1_approx, curr_head + l1_approx)
        curr_errors = curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:])
        res_errors = None if res_errors is None else res_errors.view(res_head.shape[0], -1, *res_head.shape[1:])
        return curr_head, curr_errors, res_head, res_errors, bounds

    def forward(self, head, errors, end_layer, compute_bounds=False, adv_errors=None, return_prop=False):
        curr_head, curr_errors, res_head, res_errors, bounds = self.propagate(self.relaxed_blocks, head, errors,
                                                                              compute_bounds, self.n_rand_proj,
                                                                              end_layer, adv_errors)
        if return_prop:
            return curr_head, curr_errors, res_head, res_errors
        if compute_bounds:
            return bounds

        latent = curr_head + curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:]).sum(dim=1)
        if res_head is not None:
            res_latent = res_head + res_errors.view(res_head.shape[0], -1, *res_head.shape[1:]).sum(dim=0)
        else:
            res_latent = None

        return latent, res_latent