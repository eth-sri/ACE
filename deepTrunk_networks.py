import torch
import re
from layers import Linear, Sequential, Entropy, add_bounds
from networks import SeqNet
from relaxed_networks import CombinedNetwork
from utils import get_net, init_slopes, load_net_state
from zonotope import HybridZonotope, clamp_image


class MyDeepTrunkNet(torch.nn.Module):
    def __init__(self, device, args, dataset, trunk_net, input_size, input_channel, n_class, n_branches, gate_type,
                 branch_net_names, gate_net_names, evalFn, lossFn):
        super(MyDeepTrunkNet, self).__init__()
        self.dataset = dataset
        self.input_size = input_size
        self.input_channel = input_channel
        self.n_class = n_class
        self.gate_type = gate_type
        self.n_branches = n_branches
        self.trunk_net = trunk_net
        self.evalFn = evalFn
        self.lossFn = lossFn

        assert gate_type in ["entropy", "net"], f"Unknown gate mode: {gate_type:s}"

        self.exit_ids = [-1] + list(range(n_branches))

        self.threshold = {exit_idx: args.gate_threshold for exit_idx in self.exit_ids[1:]}
        self.gate_nets = {}
        self.branch_nets = {}

        if len(branch_net_names) != n_branches:
            print("Number of branches does not match branch net names")
            branch_net_names = n_branches * branch_net_names[0:1]

        if gate_net_names is None:
            gate_net_names = branch_net_names
        elif len(gate_net_names) != n_branches:
            print("Number of branches does not match gate net names")
            gate_net_names = n_branches * gate_net_names[0:1]

        if args.load_branch_model is not None and len(args.load_branch_model) != n_branches:
            args.load_branch_model = n_branches * args.load_branch_model[0:1]
        if args.load_gate_model is not None and len(args.load_gate_model) != n_branches:
            args.load_gate_model = n_branches * args.load_gate_model[0:1]

        for i, branch_net_name in zip(range(n_branches), branch_net_names):
            exit_idx = self.exit_ids[i+1]
            self.branch_nets[exit_idx] = get_net(device, dataset, branch_net_name, input_size, input_channel, n_class,
                                                 load_model=None if args.load_branch_model is None else args.load_branch_model[i],
                                                  net_dim=args.cert_net_dim)

            if gate_type == "net":
                self.gate_nets[exit_idx] = get_net(device, dataset, gate_net_names[i], input_size, input_channel, 1,
                                                   load_model=None if args.load_gate_model is None else args.load_gate_model[i],
                                                   net_dim=args.cert_net_dim)
            else:
                self.gate_nets[exit_idx] = SeqNet(Sequential(*[*self.branch_nets[exit_idx].blocks, Entropy(n_class, low_mem=True, neg=True)]))
                self.gate_nets[exit_idx].determine_dims(torch.randn((2, input_channel, input_size, input_size), dtype=torch.float).to(device))
                init_slopes(self.gate_nets[exit_idx], device, trainable=False)

            self.add_module("gateNet_{}".format(exit_idx), self.gate_nets[exit_idx])
            self.add_module("branchNet_{}".format(exit_idx), self.branch_nets[exit_idx])

        if args.load_model is not None:
            old_state = self.state_dict()
            load_state = torch.load(args.load_model)
            if args.cert_net_dim is not None and not ("gateNet_0.blocks.layers.1.mean" in load_state.keys()): # Only change keys if loading from a non mixed resolution to mixed resolution
                new_dict = {}
                for k in load_state.keys():
                    if k.startswith("trunk"):
                        new_k = k
                    else:
                        k_match = re.match("(^.*\.layers\.)([0-9]+)(\..*$)", k)
                        new_k = "%s%d%s" % (k_match.group(1), int(k_match.group(2)) + 1, k_match.group(3))
                    new_dict[new_k] = load_state[k]
                load_state.update(new_dict)

            # LiRPA requires parameters to have zero batch dimension. This makes old models compatible
            for k, v in load_state.items():
                if k.endswith("mean") or k.endswith("sigma"):
                    if k in old_state:
                        load_state.update({k: v.reshape(old_state[k].shape)})

            old_state.update({k:v.view(old_state[k].shape) for k,v in load_state.items() if
                              k in old_state and (
                              (k.startswith("trunk") and args.load_trunk_model is None)
                              or (k.startswith("gate") and args.load_gate_model is None)
                              or (k.startswith("branch") and args.load_branch_model is None))})
            missing_keys, extra_keys = self.load_state_dict(old_state, strict=False)
            assert len([x for x in missing_keys if "gateNet" in x or "branchNet" in x]) == 0
            print("Whole model loaded from %s" % args.load_model)

            ## Trunk and branch nets have to be loaded after the whole model
            if args.load_trunk_model is not None:
                load_net_state(self.trunk_net, args.load_trunk_model)

        if (args.load_model is not None or args.load_gate_model is not None) and args.gate_feature_extraction is not None:
            for i, net in enumerate(self.gate_nets.values()):
                extraction_layer = [ii for ii in range(len(net.blocks)) if isinstance(net.blocks[ii],Linear)]
                extraction_layer = extraction_layer[-min(len(extraction_layer),args.gate_feature_extraction)]
                net.freeze(extraction_layer-1)

        self.trunk_cnet = None
        self.gate_cnets = {k: None for k in self.gate_nets.keys()}
        self.branch_cnets = {k: None for k in self.branch_nets.keys()}

    @staticmethod
    def get_deepTrunk_net(args, device, lossFn, evalFn, input_size, input_channel, n_class, trunk_net=None):
        if trunk_net is None:
            if args.net != "None":
                trunk_net = get_net(device, args.dataset, args.net, input_size, input_channel, n_class,
                                                 load_model=args.load_trunk_model)
        specNet = MyDeepTrunkNet(device, args, args.dataset, trunk_net, input_size, input_channel, n_class,
                                 args.n_branches, args.gate_type, args.branch_nets, args.gate_nets, evalFn, lossFn)

        specNet.add_cnets(device, lossFn, evalFn, args.n_rand_proj)
        return specNet

    def add_cnets(self, device, lossFn, evalFn, n_rand_proj, balance_factor=1):
        c = torch.tensor(balance_factor, dtype=torch.float, device=device)
        if self.gate_type == "net":
            def lossFn_gate(x, y):
                if x.dim()>1:
                    return torch.nn.functional.binary_cross_entropy_with_logits(x.squeeze(dim=1), y.float(), pos_weight=c, reduction='none')
                else:
                    return torch.nn.functional.binary_cross_entropy_with_logits(x, y.float(), pos_weight=c, reduction='none')
        else:
            def lossFn_gate(x, y):
                return (x.squeeze() * (y <= 0.) - x.squeeze() * (y > 0.)*c)

        self.trunk_cnet = CombinedNetwork.get_cnet(self.trunk_net, device, lossFn, evalFn, n_rand_proj, no_r_net=True, input_channels=self.input_channel)
        for exit_idx in self.exit_ids[1::]:
            def evalFn_gate(x): return (x.squeeze(dim=1) >= self.threshold[exit_idx]).int()

            self.gate_cnets[exit_idx] = CombinedNetwork.get_cnet(self.gate_nets[exit_idx], device, lossFn_gate,
                                                                 evalFn_gate, n_rand_proj,
                                                                 threshold_min=self.threshold[exit_idx])
            self.branch_cnets[exit_idx] = CombinedNetwork.get_cnet(self.branch_nets[exit_idx], device, lossFn, evalFn,
                                                                   n_rand_proj)

    def get_adv_loss(self, inputs, targets, adv_attack_test, mode="mixed"):
        curr_head, A_0 = clamp_image(inputs, adv_attack_test.eps)
        x_out, x_gate, selected, exit_ids = self(inputs)
        with torch.enable_grad():
            adv_input = inputs.clone()
            adv_loss = torch.zeros_like(targets).float()
            adv_found = torch.zeros_like(targets).bool()
            adv_errors = torch.FloatTensor(curr_head.shape).unsqueeze(1). \
                to(curr_head.device).uniform_(-1, 1).requires_grad_(True)
            select_mask_adv = torch.zeros_like(torch.stack(selected, dim=1))
            select_out_max = -1000*torch.ones_like(x_gate)
            select_out_min = 1000 * torch.ones_like(x_gate)
            for i, exit_idx in enumerate(self.exit_ids[1:] + [-1]):
                branch_net = self.trunk_net if exit_idx == -1 else self.branch_nets[exit_idx]

                gate_nets = [self.gate_cnets[j] for j in range(self.n_branches) if j <= i]
                gate_target = torch.zeros((len(targets), self.n_branches), dtype=torch.float, device=targets.device)
                if exit_idx != -1:
                    gate_target[:, i] = 1

                for it in range(adv_attack_test.n_steps+1):
                    curr_errors = A_0.unsqueeze(1) * adv_errors
                    adv_latent = curr_head + curr_errors.view(curr_head.shape[0], -1, *curr_head.shape[1:]).sum(dim=1)

                    if it == adv_attack_test.n_steps:
                        adv_outs, adv_gate_outs, exit_mask, _ = self.forward(adv_latent)
                        select_out_max = adv_gate_outs.max(select_out_max)
                        select_out_min = adv_gate_outs.min(select_out_min)

                        select_mask_adv = select_mask_adv.__or__(torch.stack(exit_mask, dim=1))
                        if branch_net is None:
                            break
                        adv_branch = adv_outs[:,i]
                        ce_loss = self.lossFn(adv_branch, targets)
                        break

                    if mode == "reachability" or (branch_net is None):
                        ce_loss = 0
                    else:
                        adv_branch = branch_net(adv_latent)
                        ce_loss = self.lossFn(adv_branch, targets)
                    if not mode == "classification":
                        for j, gate_net in enumerate(gate_nets):
                            adv_gate = gate_net.net(adv_latent)
                            ce_loss_gate = gate_net.lossFn(adv_gate, gate_target[:,j])
                            weight = torch.nn.functional.softplus(
                                     2 * (adv_gate.view(-1) - gate_net.threshold_min)
                                     * (1 - 2 * gate_target[:, j])
                                     + 0.5).detach()
                            ce_loss -= ce_loss_gate * weight

                    self.zero_grad()
                    ce_loss.sum(dim=0).backward()
                    adv_errors.data = torch.clamp(adv_errors.data + adv_attack_test.step_size * adv_errors.grad.sign(), -1, 1)
                    adv_errors.grad.zero_()
                adv_input = torch.where(((adv_loss < ce_loss).__and__(exit_mask[i])).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1), adv_latent, adv_input)
                adv_found = (adv_loss < ce_loss).__and__(exit_mask[i]).__or__(adv_found)
                adv_loss = torch.where((adv_loss < ce_loss).__and__(exit_mask[i]), ce_loss, adv_loss)
            adv_input = torch.where(adv_found.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1), adv_input, adv_latent)
            adv_latent = adv_input.detach()
            adv_outs, _, exit_mask, adv_branches = self.forward(adv_latent)
            if self.trunk_net is None:
                exit_mask[-2] = exit_mask[-2].__or__(exit_mask[-1])
                exit_mask = exit_mask[:-1]
            adv_loss = self.lossFn(adv_outs[torch.stack(exit_mask, dim=1)], targets)
            adv_ok = targets.eq(self.evalFn(adv_outs[torch.stack(exit_mask,dim=1)])).detach()
        if mode == "reachability":
            return select_mask_adv, select_out_min, select_out_max
        return adv_loss, adv_ok, adv_outs, adv_latent, adv_branches

    def forward(self, x):
        x_out = []
        x_gate = []
        selected = []
        remaining = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        exit_ids = torch.zeros(x.size(0), dtype=torch.int, device=x.device)

        for exit_idx in self.exit_ids[1:]:

            x_out += [self.branch_nets[exit_idx].forward(x=x.clone())]
            x_gate += [self.gate_nets[exit_idx].forward(x=x.clone())]

            selected += [(x_gate[-1].squeeze(dim=1) >= self.threshold[exit_idx]).__and__(remaining)]

            exit_ids[selected[-1]] = exit_idx
            remaining[selected[-1]] = False

        exit_ids[remaining] = -1
        selected += [remaining]

        if self.trunk_net is not None:
            x_out += [self.trunk_net.forward(x=x)]

        x_out = torch.stack(x_out, dim=1)
        x_gate = torch.stack(x_gate, dim=1)

        return x_out, x_gate, selected, exit_ids

    def reset_bounds(self):
        if self.trunk_net is not None:
            self.trunk_net.reset_bounds()
        for gate_net in self.gate_nets.values():
            gate_net.reset_bounds()
        for branch_net in self.branch_nets.values():
            branch_net.reset_bounds()

    def compute_bounds(self, inputs, domain, eps, compute_for_trunk=True):
        with torch.no_grad():
            for exit_idx, gate_net in self.gate_nets.items():
                if self.gate_nets[exit_idx].transform is not None and self.gate_nets[exit_idx].transform.consolidate_errors:
                    abs_gate = HybridZonotope.construct_from_noise(inputs, eps, "box")
                    abs_gate.domain = domain
                else:
                    abs_gate = HybridZonotope.construct_from_noise(inputs, eps, domain)
                for j, layer in enumerate(gate_net.blocks):
                    lidx = "g%d_%d" % (exit_idx, j)
                    add_bounds(lidx, abs_gate, layer=layer)
                    abs_gate = layer(abs_gate) if not isinstance(layer,Entropy) else layer(abs_gate, bounds=True)
            del abs_gate

            last_exit = -1
            for exit_idx, branch_net in self.branch_nets.items():
                if branch_net.transform is not None and branch_net.transform.consolidate_errors:
                    abs_branch = HybridZonotope.construct_from_noise(inputs, eps, "box")
                    abs_branch.domain = domain
                else:
                    abs_branch = HybridZonotope.construct_from_noise(inputs, eps, domain)

                for k, layer in enumerate(branch_net.blocks):
                    lidx = "b%d_%d" % (exit_idx, k)
                    add_bounds(lidx, abs_branch, layer=layer)
                    abs_branch = layer(abs_branch) if not isinstance(layer,Entropy) else layer(abs_gate, bounds=True)
                del abs_branch
                last_exit = exit_idx

            if compute_for_trunk and self.trunk_net is not None:
                if self.trunk_net.transform is not None and self.trunk_net.transform.consolidate_errors:
                    abs_trunk = HybridZonotope.construct_from_noise(inputs, eps, "box")
                    abs_trunk.domain = domain
                else:
                    abs_trunk = HybridZonotope.construct_from_noise(inputs, eps, domain)

                for j, layer in enumerate(self.trunk_net.blocks[last_exit + 1:]):
                    lidx = "%d" % (j + last_exit + 1)
                    add_bounds(lidx, abs_trunk, layer=layer)
                    abs_trunk = layer(abs_trunk)
                del abs_trunk
