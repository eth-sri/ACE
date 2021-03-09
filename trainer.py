import torch
import torch.optim as optim
import os
import csv
from utils import AdvAttack, get_lp_loss, my_cauchy
from tqdm import tqdm
from zonotope import clamp_image, HybridZonotope
from tabulate import tabulate
from layers import Entropy


def test(device, args, cnet, test_loader, layers, stats=None, ind_class=False, log_ind=True, post_fix=""):
    cnet.net.eval()

    # Add adversarial evaluation
    if not -1 in layers:
        layers = [-1]+layers

    # Initialize scores
    test_nat_loss, test_nat_ok, n_batches = 0, 0, 0
    test_pgd_loss, test_pgd_ok = {}, {}
    nat_y, adv_y, y = [], [], []
    sample_weights_list = []
    threshold = []
    for layer_idx in layers:
        test_pgd_loss[layer_idx], test_pgd_ok[layer_idx] = 0, 0

    # Create AdvAttack object
    adv_attack_test = AdvAttack(eps=args.test_eps, n_steps=args.test_att_n_steps,
                                 step_size=args.test_att_step_size, adv_type="pgd")

    # Initialize sample-wise logging
    if log_ind:
        # create new loader to ensure shuffle is turned off and no samples are dropped
        test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=test_loader.batch_size,
                                    shuffle=False, num_workers=test_loader.num_workers, pin_memory=True,
                                    drop_last=False)
        log_file= os.path.join(args.model_dir, "test_log%s.csv"%post_fix)
        csv_log = open(log_file, mode='w')
        print("Writing log to %s" % log_file)
        log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(["img_id", "label", "y", "adv_y", "nat_ok"] + ["pgd_ok_%d" % (x) for x in layers])
    else:
        log_writer = None

    pbar = tqdm(test_loader, dynamic_ncols=True)
    n = 0

    for it, data in enumerate(pbar):
        if args.debug and it == 50:
            pbar.close()
            break

        # If sample weights are part of the dataset load those as well
        if len(data) == 2:
            inputs, targets = data
            sample_weights = torch.ones(targets.size(), dtype=torch.float)
        else:
            inputs, targets, sample_weights = data

        inputs, targets = inputs.to(device), targets.to(device)
        n_batches += 1

        cnet.net.reset_bounds()

        nat_loss, nat_ok, nat_y_b, adv_latent_loss, adv_ok, adv_y_b, _, _, stable_losses, threshold_batch = \
            cnet(inputs, targets, args, None, adv_attack_test, layer_ids=layers, is_train=False, domains=None)

        nat_y.append(nat_y_b.detach().cpu())
        adv_y.append({k:v.detach().cpu() for k,v in adv_y_b.items() if k==-1})
        y.append(targets.cpu())
        sample_weights_list.append(sample_weights)
        threshold.append(threshold_batch.detach())

        test_nat_loss += nat_loss.mean().item()
        test_nat_ok += nat_ok.mean().item()
        for layer_idx in layers:
            test_pgd_loss[layer_idx] += adv_latent_loss[layer_idx].mean().item()
            test_pgd_ok[layer_idx] += adv_ok[layer_idx].mean().item()

        if log_writer is not None:
            for n_i in range(len(targets)):
                log_writer.writerow([n+n_i, targets[n_i].item(), nat_y_b[n_i].item(), adv_y_b[-1][n_i].item(), nat_ok[n_i].int().item()] + [adv_ok[x][n_i].int().item() for x in layers])

        if stats is not None:
            stats.report('nat_loss/test', nat_loss.mean().item())
            stats.report('nat_ok/test', nat_ok.mean().item())
            stats.report('nat_y/test', nat_y_b.float().mean().item())
            stats.report('y/test', targets.float().mean().item())
            stats.report('adv_y/test', adv_y_b[-1].float().mean().item())

            for layer_idx in adv_latent_loss.keys():
                if stats is not None:
                    stats.report('adv_loss/test/%d' % layer_idx, adv_latent_loss[layer_idx].mean().item())
                    stats.report('adv_ok/test/%d' % layer_idx, adv_ok[layer_idx].mean().item())

            for layer_idx in stable_losses.keys():
                stats.report('stable_loss/%d' % layer_idx, stable_losses[layer_idx].mean().item())

        abs_pgd_ok_str = ', '.join(['%d: %.4f' % (layer_idx, test_pgd_ok[layer_idx]/n_batches) for layer_idx in layers])
        abs_pgd_loss_str = ', '.join(['%d: %.4f' % (layer_idx, test_pgd_loss[layer_idx]/n_batches) for layer_idx in layers])
        pbar.set_description('[V] nat_loss=%.4f, nat_ok=%.4f, pgd_loss={%s}, pgd_ok={%s}' % (
            test_nat_loss/n_batches, test_nat_ok/n_batches, abs_pgd_loss_str, abs_pgd_ok_str))
        n += len(targets)

    if log_writer is not None:
        log_writer.writerow(["total", "", "", "", test_nat_ok/n_batches] + [test_pgd_ok[x]/n_batches for x in layers])
        csv_log.close()

    nat_y = torch.cat(nat_y, dim=0)
    adv_y = {k: torch.cat([x[k] for x in adv_y], dim=0) for k in adv_y[0].keys()}
    y = torch.cat(y, dim=0)
    sample_weights_list = torch.cat(sample_weights_list, dim=0)
    if ind_class:
        return test_nat_loss/n_batches, test_nat_ok/n_batches, test_pgd_loss[layers[0]]/n_batches, test_pgd_ok[layers[0]]/n_batches, y, nat_y, adv_y[-1], sample_weights_list
    return test_nat_loss/n_batches, test_nat_ok/n_batches, test_pgd_loss[layers[0]]/n_batches, test_pgd_ok[layers[0]]/n_batches


def compute_bounds(net, inputs, domain, eps, bounds=None, layer_idx=None, lidx_prefix=""):
    if bounds is None:
        bounds = {}
    if layer_idx is None:
        layer_idx = len(net.blocks)-1

    if net.transform is not None and net.transform.consolidate_errors:
        abs_curr = HybridZonotope.construct_from_noise(inputs, eps, "box")
        abs_curr.domain = domain
    else:
        abs_curr = HybridZonotope.construct_from_noise(inputs, eps, domain)

    for j, layer in enumerate(net.blocks[:layer_idx+1]):
        add_bounds(bounds, "%s%d" % (lidx_prefix,j) , abs_curr, layer)
        abs_curr = layer(abs_curr) if not isinstance(layer, Entropy) else layer(abs_curr, bounds=True)
    return bounds


def add_bounds(bounds, lidx, zono, layer=None):
    lb_new, ub_new = zono.concretize()
    bounds[lidx] = (lb_new, ub_new)
    if layer is not None:
        if layer.bounds is not None:
            lb_old, ub_old = layer.bounds
            lb_new, ub_new = torch.max(lb_old, lb_new).detach(), torch.min(ub_old, ub_new).detach()
        layer.bounds = (lb_new, ub_new)
    return bounds


def train(device, epoch, args, relu_idx, layers, cnets, eps_sched, kappa_sched, opt, train_loader, lr_scheduler,
          relu_ids, stats=None, relu_stable=None, relu_stable_protected=0, net_weights=[1], beta_sched=None):

    # Combine losses of more than one network
    if not isinstance(cnets,list):
        cnets = [cnets]
    if len(net_weights) != len(cnets):
        net_weights = len(cnets) * [1]
    net_weights = torch.Tensor(net_weights).to(device).float()
    net_weights = net_weights/net_weights.sum()
    for cnet in cnets: cnet.train()

    nets = [cnet.net for cnet in cnets]

    train_mode = args.train_mode
    layer_id = layers[relu_idx]
    layer_prev_id = layers[relu_idx-1]
    nat_factor = args.nat_factor

    for net in nets: net.freeze(layer_id)

    # If ReLU stable loss is used, calculate bounds up to the corresponding layer, otherwise only to the current layer
    if relu_stable is not None and layer_id < relu_ids[-relu_stable_protected-1]:
        stable_idx = [x for x in relu_ids if x > layers[relu_idx]]
        stable_idx = stable_idx[:min(args.relu_stable_cnt, len(stable_idx))]
    else:
        stable_idx = [layer_id] if layer_id >= 0 else None

    pbar = tqdm(train_loader, dynamic_ncols=True)

    for it, data in enumerate(pbar):
        if args.debug and it==50:
            pbar.close()
            kappa_sched.advance_time(args.train_batch*(len(pbar)-50))
            eps_sched.advance_time(args.train_batch*(len(pbar)-50))
            break

        if len(data) == 2:
            inputs, targets = data
            sample_weights = torch.ones(targets.size(), dtype=float)
        else:
            inputs, targets, sample_weights = data

        inputs, targets, sample_weights = inputs.to(device), targets.to(device), sample_weights.to(device)
        eps, kappa = eps_sched.get(), kappa_sched.get()
        beta = 1 if beta_sched is None else beta_sched.get()
        curr_lr = opt.param_groups[0]['lr']

        for net in nets: net.reset_bounds()

        if "natural" not in train_mode:
            adv_attack_train = AdvAttack(eps=eps, n_steps=args.train_att_n_steps,
                                         step_size=args.train_att_step_size, adv_type=args.adv_type)
        else:
            adv_attack_train = None

        if "COLT" in train_mode:
            layer_ids = [layer_prev_id, layer_id]
        elif "natural" in train_mode:
            layer_ids = [-2, -2]
            nat_factor = 1.
        elif "adv" in train_mode:
            layer_ids = [-2, -1]
        elif "diffAI" in train_mode:
            layer_ids = [-2, -2]
            nat_factor = kappa * args.nat_factor + (1-kappa)
        else:
            assert False, "Unknown train mode"

        nat_loss, nat_ok, nat_y_b, adv_latent_loss, adv_ok, adv_y_b, abs_loss, abs_ok, stable_losses, _ = \
        cnets[0](inputs, targets, args, stable_idx, adv_attack_train, layer_ids, is_train=True,
             relu_stable_type=args.relu_stable_type,
             train_mode=train_mode, domains=args.train_domain, beta=beta)

        if args.gate_target_type == "nat":
            targets_gate = nat_ok.clone().float()
        elif args.gate_target_type == "equal":
            adv_ok.copy()
            targets_gate.update(abs_ok)
        else:
            raise RuntimeError(f"gate_target_type {args.gate_target_type:%s} not recognized.")

        cnets_out = [cnet(inputs, targets_gate, args, stable_idx, adv_attack_train, layer_ids, is_train=True,
                 relu_stable_type=args.relu_stable_type,
                 train_mode=train_mode, domains=args.train_domain, kappa=kappa) for cnet in cnets[1:]]

        if len(cnets_out) > 0:
            nat_loss = (torch.stack(([nat_loss]+[cnet_out[0] for cnet_out in cnets_out]),dim=0)*net_weights.unsqueeze(dim=1)).sum(dim=0)
            adv_latent_loss = {key: (torch.stack(([adv_latent_loss[key]]+[cnet_out[3][key] for cnet_out in cnets_out]),dim=0)*net_weights.unsqueeze(dim=1)).sum(dim=0) for key in cnets_out[0][3].keys()}
            abs_loss = {key: (torch.stack(([abs_loss[key]]+[cnet_out[6][key] for cnet_out in cnets_out]),dim=0)*net_weights.unsqueeze(dim=1)).sum(dim=0) for key in cnets_out[0][6].keys()}
            stable_losses = {key: (torch.stack(([stable_losses[key]]+[cnet_out[8][key] for cnet_out in cnets_out]),dim=0)*net_weights).sum(dim=0) for key in cnets_out[0][8].keys()}

        if stats is not None:
            stats.report('nat_factor', nat_factor)
            stats.report('nat_loss/train', nat_loss.mean().item())
            stats.report('nat_ok/train', nat_ok.mean().item())
            stats.report('nat_y/train', nat_y_b.float().mean().item())
            stats.report('y/train', targets.float().mean().item())
            for layer_idx in adv_latent_loss.keys():
                if stats is not None:
                    stats.report('adv_loss/train/%d' % layer_idx, adv_latent_loss[layer_idx].mean().item())
                    stats.report('adv_ok/train/%d' % layer_idx, adv_ok[layer_idx].mean().item())
                    if layer_idx >-2:
                        stats.report('adv_y/train/%d' % layer_idx, adv_y_b[layer_idx].float().mean().item())
            for domain in abs_loss.keys():
                if stats is not None:
                    stats.report('abs_loss/train/%s' % domain, abs_loss[domain].mean().item())
                    stats.report('abs_ok/train/%s' % domain, abs_ok[domain].mean().item())
            if stable_idx is not None and relu_stable is not None:
                for layer_idx in stable_idx:
                    stats.report('stable_loss/%d' % layer_idx, stable_losses[layer_idx].mean().item())

        if "natural" in train_mode:
            adv_loss = 0
        elif "diffAI" in train_mode:
            adv_loss = 0
            for domain in abs_loss.keys():
                adv_loss += args.train_domain_weights[domain] * abs_loss[domain]
        else:
            latent_loss_weights = {}
            latent_loss_weights[layer_ids[0]] = 1-kappa
            latent_loss_weights[layer_ids[1]] = kappa
            adv_loss = 0
            for loss_layer_idx in layer_ids:
                adv_loss += latent_loss_weights[loss_layer_idx] * adv_latent_loss[loss_layer_idx]

        stable_loss = 0
        if stable_idx is not None and relu_stable is not None:
            for layer_idx in stable_idx:
                stable_loss += relu_stable * stable_losses[layer_idx]

        l1_loss = get_lp_loss(nets[0].blocks, p=1, input_size=inputs.size(-1), scale_conv=args.reg_scale_conv)
        l2_loss = get_lp_loss(nets[0].blocks, p=2, input_size=inputs.size(-1), scale_conv=args.reg_scale_conv)
        reg_loss = args.l1_reg * l1_loss
        reg_loss += args.l2_reg * l2_loss

        tot_loss = ((nat_factor * nat_loss
                   + (1 - nat_factor) * adv_loss
                   + stable_loss)
                   * sample_weights).mean() \
                   + reg_loss

        opt.zero_grad()
        tot_loss.backward()
        opt.step()

        if isinstance(lr_scheduler, optim.lr_scheduler.CyclicLR):
            lr_scheduler.step()

        if stats is not None:
            stats.report('lr', curr_lr)
            stats.report('kappa', kappa)
            stats.report('eps', eps)
            stats.report('tot_loss/train', tot_loss.item())

        msg_cert = ", ".join(['%s: %.4f' % (domain, stats.get('abs_ok/train/%s'%domain)) for domain in abs_ok.keys()])
        pbar.set_description('[T] epoch=%d, lr=%.2e, eps=%.5f, kappa=%.3f, loss=%.4f, nat_ok=%.4f, adv_ok=%.4f, abs_ok={%s}' % (
            epoch, curr_lr, eps, kappa, stats.get('tot_loss/train'),
            stats.get('nat_ok/train'),
            -1 if layer_id not in adv_latent_loss or layer_id < -1 else stats.get('adv_ok/train/%d' % layers[relu_idx]),
            msg_cert))
        eps_sched.advance_time(args.train_batch)
        kappa_sched.advance_time(args.train_batch)


def get_opt(net, opt, lr, lr_step, lr_factor, n_epochs, train_loader, lr_scheduler_type, lr_scheduler=None, pct_up=0.5, fixup=False):
    if opt == 'adam':
        opt = optim.Adam(net.parameters(), lr=lr)
    else:
        if fixup:
            params_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
            params_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
            params_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
            opt = optim.SGD(
                [{'params': params_bias, 'lr': lr},
                 {'params': params_scale, 'lr': lr},
                 {'params': params_others, 'lr': lr}],
                lr=lr,
                momentum=0.9)
        else:
            opt = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    if lr_scheduler is None:
        if lr_scheduler_type == 'step_lr':
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_factor)
        else:
            lr_steps = n_epochs * len(train_loader)
            lr_scheduler = optim.lr_scheduler.CyclicLR(opt, base_lr=0.0, max_lr=lr, step_size_up=pct_up*lr_steps, step_size_down=(1-pct_up)*lr_steps)
    else:
        lr_scheduler.optimizer = opt
    return opt, lr_scheduler


def diffAI_cert(device, args, cnet, data_loader, stats=None, cert_mode="acc", log_ind=False, DDP_enabled=False,
                exact=True, ind_class=False, epoch=None, domains=None, print_table=True, post_fix="", threshold_min=0):
    cnet.eval()
    if domains is None or (isinstance(domains,list) and None in domains): domains = args.cert_domain
    tot_ver = {domain: 0 for domain in domains}
    tot_ver_corr = {domain: 0 for domain in domains}
    tot_corr = 0
    tot_adv_corr = 0
    tot_samples = 0
    img_id = 0
    threshold = {domain: [] for domain in domains}
    threshold_adv = []
    nat_y = []
    adv_y = []
    y = []
    sample_weights_list = []
    if not domains == ["zono"]:
        exact = True

    net = cnet.module.net if DDP_enabled else cnet.net
    relaxed_net = cnet.module.relaxed_net if DDP_enabled else cnet.relaxed_net

    adv_attack_test = AdvAttack(eps=args.test_eps, n_steps=args.test_att_n_steps,
                                step_size=args.test_att_step_size, adv_type="pgd")

    n_class = net.blocks[-1].out_features

    if log_ind and exact:
        csv_log = open(os.path.join(args.model_dir, "cert_log%s.csv"%post_fix), mode='w')
        log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cert_domain_list = [x for x in (["ver_%s" % domain for domain in domains]+["ver_ok_%s" % domain for domain in domains])]
        cert_threshold_list = ["cert_threshold_%s" % domain for domain in domains]
        log_writer.writerow(["img_id", "label", "nat_ok", "pgd_ok"] + cert_domain_list + ["adv_threshold"] + cert_threshold_list)

    pbar = tqdm(data_loader, dynamic_ncols=True)

    for it, data in enumerate(pbar):
        if len(data)==2:
            inputs, targets = data
            sample_weights = torch.ones(size=targets.size(),dtype=torch.float)
        else:
            inputs, targets, sample_weights = data
        if args.debug and it == 50:
            pbar.close()
            break

        inputs, targets = inputs.to(device), targets.to(device)
        tot_samples += inputs.size()[0]

        if cert_mode == "acc":
            targets = targets
        elif cert_mode == "robust":
            targets = cnet.evalFn(net(inputs)).detach()
        elif isinstance(cert_mode, int):
            targets = int(cert_mode)*torch.ones_like(targets)
        else:
            assert False, "cert_mode not recognized"

        _, nat_ok, nat_y_batch, _, adv_ok, adv_y_batch, _, _, _, threshold_adv_batch = \
            cnet(inputs, targets, args, None, adv_attack_test, [-1], is_train=False, domains=None)
        adv_ok = adv_ok[-1].detach()
        nat_y.append(nat_y_batch.detach())
        adv_y.append({k:v.detach() for k,v in adv_y_batch.items()})
        if cnet.targetFn_test is not None:
            y.append(cnet.targetFn_test(targets))
        else:
            y.append(targets)
        sample_weights_list.append(sample_weights)

        tot_corr += nat_ok.int().sum().item()
        tot_adv_corr += adv_ok.int().sum().item()

        if exact:
            threshold_batch = {domain:torch.zeros(targets.size(), dtype=torch.float, device=device) for domain in domains}
            for n in range(inputs.shape[0]):
                cert_list = []
                threshold_list = []
                for domain in domains:
                    if "zono" not in domain or domain == "hbox":
                        # If box is only domain run certification for whole batch at once
                        target = targets
                        input_img = inputs
                    else:
                        img_id += 1
                        target = targets[n:n + 1]
                        input_img = inputs[n:n + 1]
                    ver_corr, threshold_n, _ = cnet.get_abs_loss(input_img, target, adv_attack_test.eps, domain, threshold_min, beta=1)

                    if not cert_mode == "acc":
                        # for binary classification
                        assert ((target == 1).__or__(target == 0)).all()
                        threshold_n = (threshold_n * (-1+2*target))

                    tot_ver[domain] += 0#ver.int().sum().item()
                    tot_ver_corr[domain] += ver_corr.int().sum().item()
                    cert_list += [torch.zeros_like(ver_corr).int().detach(), ver_corr.int().detach()]#[ver.int().detach(), ver_corr.int().detach()]

                    if not "zono" in domain or domain =="hbox":
                        threshold_batch[domain] = threshold_n.detach()
                        threshold_list += [threshold_batch[domain]]
                    else:
                        threshold_batch[domain][n] = threshold_n.item() if isinstance(threshold_n, torch.Tensor) else threshold_n
                        threshold_list += [threshold_batch[domain][n].item()]

                if "zono" not in domain or domain =="hbox":
                    cert_list = torch.stack(cert_list, dim=1)
                    threshold_list = torch.stack(threshold_list, dim=1)
                    if log_ind and (device == 0 or not DDP_enabled):
                        for nn in range(inputs.shape[0]):
                            img_id += 1
                            log_writer.writerow([img_id, target[nn].item(), nat_ok[nn].int().item(), adv_ok[nn].int().item()]
                                            + cert_list[nn].tolist() + [threshold_adv_batch[nn].item()] + threshold_list[nn].tolist())
                    break
                else:
                    if isinstance(cert_list[0], torch.Tensor):
                        cert_list = torch.cat(cert_list, dim=0).tolist()
                    if log_ind and (device == 0 or not DDP_enabled):
                        log_writer.writerow([img_id, target.item(), nat_ok[n].int().item(), adv_ok[n].int().item()]
                                            + cert_list + [threshold_adv_batch[n].item()] + threshold_list)
            for domain in domains:
                threshold[domain].append(threshold_batch[domain])

        else: # approximate certifcation used during training with zono
            assert len(domains) == 1 and "zono" in domains[0], "Approximate certificaiton is only supported for the zono domain"
            relaxed_net.eval()
            curr_head, A_0 = clamp_image(inputs, adv_attack_test.eps)
            curr_errors = A_0.unsqueeze(1) * my_cauchy(inputs.shape[0], args.n_rand_proj, *inputs.shape[1:]).to(
                curr_head.device)
            assert not torch.isinf(curr_errors).any()
            end_layer = len(net.blocks)
            curr_head, curr_errors, res_head, res_errors = relaxed_net.forward(curr_head, curr_errors, compute_bounds=True,
                                                                           end_layer=end_layer, return_prop=True)
            if n_class > 1:
                c = torch.stack([net.get_c_mat(n_class, x, device) for x in targets], dim=0)
                curr_head = (curr_head.unsqueeze(dim=1) * c.to(device)).sum(dim=2)
                curr_errors = (curr_errors.unsqueeze(dim=2) * c.to(device).unsqueeze(dim=1)).sum(dim=3)

            l1_approx = 0
            for i in range(0, curr_errors.shape[1], relaxed_net.n_rand_proj):
                l1_approx += torch.median(curr_errors[:, i:i + relaxed_net.n_rand_proj].abs(), dim=1)[0]
            lb = (curr_head - l1_approx).detach()
            ub = (curr_head + l1_approx).detach()
            del curr_head, curr_errors, res_errors, res_head

            if n_class == 1:
                threshold_batch = lb
                threshold_batch[targets == 0] = ub[targets == 0]
                threshold_batch = (threshold_batch.squeeze(dim=1) * (-1 + 2 * targets.view(-1)))
            else:
                threshold_batch = lb[torch.stack([x != targets for x in range(lb.size(1))],dim=1)].view(-1,n_class-1).min(dim=1)[0]

            ver_corr = threshold_batch > threshold_min
            ver = ver_corr
            tot_ver[domains[0]] += ver.int().sum().item()
            tot_ver_corr[domains[0]] += ver_corr.int().sum().item()
            threshold[domains[0]].append(threshold_batch)

        msg_nat = '%.4f' % (tot_corr/tot_samples)
        msg_adv = '%.4f' % (tot_adv_corr/tot_samples)
        msg_ver = ", ".join(['%s: %.4f' % (domain, tot_ver[domain]/tot_samples) for domain in domains])
        msg_ver_corr = ", ".join(['%s: %.4f' % (domain, tot_ver_corr[domain]/tot_samples) for domain in domains])
        pbar.set_description('[C] tot_nat= %s, tot_adv= %s, tot_ver= {%s}, tot_ver_corr= {%s}' %
                             (msg_nat, msg_adv, msg_ver, msg_ver_corr))
        if not cert_mode == "acc": threshold_adv_batch = (threshold_adv_batch.squeeze(dim=1) * (-1 + 2 * targets.view(-1)))
        threshold_adv.append(threshold_adv_batch.detach())

    threshold = {domain: torch.cat(threshold[domain], dim=0) for domain in domains}
    threshold_adv = torch.cat(threshold_adv, dim=0)
    nat_y = torch.cat(nat_y, dim=0)
    adv_y = {k: torch.cat([x[k] for x in adv_y], dim=0) for k in adv_y[0].keys()}
    y = torch.cat(y, dim=0)
    sample_weights_list = torch.cat(sample_weights_list, dim=0)

    threshold_adv_max = 0
    threshold_cert_max = 0

    if log_ind and exact:
        cert_domain_list = [x for x in ([tot_ver[domain] / tot_samples for domain in domains]+[tot_ver_corr[domain] / tot_samples for domain in domains])]

        if n_class == 1 :
            log_writer.writerow(["total", "", tot_corr / tot_samples, tot_adv_corr / tot_samples] + cert_domain_list +
                                [threshold_adv_max, threshold_cert_max])
        else:
            log_writer.writerow(["total", "", tot_corr/tot_samples, tot_adv_corr/tot_samples] + cert_domain_list +
                                ["", ""])
        csv_log.close()

    if stats is not None:
        for domain in tot_ver.keys():
            stats.report('cert/acc_ver/%s' % domain, tot_ver_corr[domain]/tot_samples)
            stats.report('cert/ver/%s' % domain, tot_ver[domain]/tot_samples)
        stats.update_tb(epoch)

    if print_table:
        headers = [' ', 'acc', 'adv_acc']
        table = [
            [args.net, f'{tot_corr / tot_samples:.3f}', f'{tot_adv_corr / tot_samples:.3f}']
        ]
        for domain in domains:
            headers.append(f'ver_acc {domain}')
            table[0].append(f'{tot_ver_corr[domain]/tot_samples:.3f}')
        tab = tabulate(table, headers, tablefmt='github')
        print(tab)

    if ind_class:
        return threshold, threshold_adv, y, nat_y, adv_y[-1], sample_weights_list
    return threshold, threshold_adv

