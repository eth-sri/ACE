import time
import json
import numpy as np
import os
import torch
import torch.nn as nn
import csv
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from args_factory import get_args
from loaders import get_loaders
from utils import (Scheduler, Statistics, count_vars, write_config, AdvAttack, MyVisionDataset, get_scaled_eps, get_layers)
from tqdm import tqdm
from layers import ReLU, Linear
from trainer import train, test, get_opt, diffAI_cert
import random
from warnings import warn
from deepTrunk_networks import MyDeepTrunkNet
from zonotope import HybridZonotope


seed = 100

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=10)
np.random.seed(seed)
random.seed(seed)


def train_deepTrunk(dTNet, args, device, stats, train_loader, test_loader):
    epoch = 0
    lr = args.lr
    n_epochs = args.n_epochs
    last_gate_layer = -2

    ## Get a duplicate of the data loaders that will be changed
    train_set_spec = MyVisionDataset.from_idx(train_loader.dataset, np.ones_like(train_loader.dataset.targets).astype(bool), train=True)
    train_set_spec.set_weights(None)
    train_loader_spec = torch.utils.data.DataLoader(train_set_spec, batch_size=args.train_batch,
                                                    shuffle=~args.debug, num_workers=train_loader.num_workers, pin_memory=True, drop_last=True)
    test_set_spec = MyVisionDataset.from_idx(test_loader.dataset,
                                             np.ones_like(test_loader.dataset.targets).astype(bool), train=False)
    test_set_spec.set_weights(None)
    test_loader_spec = torch.utils.data.DataLoader(test_set_spec, batch_size=args.test_batch,
                                                   shuffle=False, num_workers=train_loader.num_workers, pin_memory=True, drop_last=False)

    ### train core-network
    if args.gate_on_trunk and args.train_trunk:
        layers = get_layers(args.train_mode, dTNet.trunk_cnet, n_attack_layers=args.n_attack_layers,
                            min_layer=-1, base_layers=False,
                            protected_layers=args.protected_layers)
        assert len(layers) > 0
        print("training trunk")
        trunk_trainability_state = dTNet.trunk_net.get_freeze_state()
        epoch = train_episode(dTNet.trunk_cnet, dTNet, device, args, lr, epoch, n_epochs, train_loader_spec,
                              test_loader_spec, -1, layers, stats=stats, eps_init=0)

    ### Start training of all certification networks
    for k, exit_idx in enumerate(dTNet.exit_ids[1::]):
        if (not args.gate_on_trunk) and train_loader_spec is not None and len(train_loader_spec) > 0 and args.load_branch_model is None:
            layers = get_layers(args.train_mode, dTNet.branch_cnets[exit_idx],
                                n_attack_layers=args.n_attack_layers,
                                min_layer=-1, base_layers=False,
                                protected_layers=args.protected_layers)

            print("Training branch %d" % exit_idx)
            net_weights = [1] if args.cotrain_entropy is None else [1 - args.cotrain_entropy, args.cotrain_entropy]
            branch_trainability_state = dTNet.branch_nets[exit_idx].get_freeze_state()
            epoch = train_episode(dTNet.branch_cnets[exit_idx]
                                  if not (dTNet.gate_type == "entropy" and args.cotrain_entropy is not None)
                                  else
                                  [dTNet.branch_cnets[exit_idx], dTNet.gate_cnets[exit_idx]], dTNet, device, args,
                                  lr, epoch, n_epochs, train_loader_spec, test_loader_spec, exit_idx, layers,
                                  stats=stats, net_weights=net_weights)
            if args.retrain_branch:
                dTNet.branch_nets[exit_idx].restore_freeze(branch_trainability_state)

        ### train gate unless entropy is used for selection
        if args.gate_type == "net":
            print("getting dataset for gate training")
            gate_train_loader, gate_test_loader = get_gated_data_loaders(args, device,
                                                                         dTNet.trunk_cnet if args.gate_on_trunk else dTNet.branch_cnets[exit_idx],
                                                                         train_loader_spec, test_loader_spec, stats,
                                                                         mode=args.gate_mode,
                                                                         exact=args.exact_target_cert)
            min_layer = -2
            if args.gate_feature_extraction is not None:
                extraction_layer = [ii for ii in range(len(dTNet.gate_nets[exit_idx].blocks)) if isinstance(dTNet.gate_nets[exit_idx].blocks[ii], Linear)]
                extraction_layer = extraction_layer[-min(len(extraction_layer), args.gate_feature_extraction)]
                min_layer = [ii for ii in range(len(dTNet.gate_nets[exit_idx].blocks)) if isinstance(dTNet.gate_nets[exit_idx].blocks[ii], ReLU) and ii<extraction_layer][-1]

            layers = get_layers(args.train_mode, dTNet.gate_cnets[exit_idx], n_attack_layers=args.n_attack_layers,
                                protected_layers=args.protected_layers, min_layer=min_layer)

            print("Training gate %d" % (exit_idx))
            epoch = train_episode(dTNet.gate_cnets[exit_idx], dTNet, device, args, lr, epoch, n_epochs,
                                  gate_train_loader, gate_test_loader, exit_idx, layers, stats,
                                  balanced_loss=args.balanced_gate_loss)

        ### train certification-network
        if args.retrain_branch:
            print("Getting datasets branch %d" % exit_idx)
            branch_train_loader, branch_test_loader = get_gated_data_loaders(args, device,
                                                                             dTNet.gate_cnets[exit_idx],
                                                                             train_loader_spec,
                                                                             test_loader_spec,
                                                                             stats, mode="branch",
                                                                             exact=args.exact_target_cert)

            layers = get_layers(args.train_mode, dTNet.branch_cnets[exit_idx],
                                n_attack_layers=args.n_attack_layers,
                                min_layer=-1, base_layers=False,
                                protected_layers=args.protected_layers)
            print("Training branch for gate %d" % exit_idx)
            net_weights = [1] if args.cotrain_entropy is None else [1 - args.cotrain_entropy, args.cotrain_entropy]
            if len(branch_train_loader) > 0:
                epoch = train_episode(dTNet.branch_cnets[exit_idx]
                                      if not (dTNet.gate_type == "entropy" and args.cotrain_entropy is not None)
                                      else [dTNet.branch_cnets[exit_idx], dTNet.gate_cnets[exit_idx]],
                                      dTNet, device, args, lr, epoch, n_epochs, branch_train_loader, branch_test_loader,
                                      exit_idx, layers, stats=stats,
                                      eps_init=0 if args.gate_on_trunk else None, net_weights=net_weights)
            else:
                raise RuntimeWarning("Certification-Network not reached by any training samples. Training skipped.")


        if args.train_trunk and (args.retrain_trunk or not args.gate_on_trunk):
            train_loader_spec, test_loader_spec = get_gated_data_loaders(args, device, dTNet.gate_cnets[exit_idx],
                                                                         train_loader_spec,
                                                                         test_loader_spec,
                                                                         stats, mode="trunk",
                                                                         threshold=0 if args.gate_type == "net" else dTNet.threshold[exit_idx],
                                                                         exact=args.exact_target_cert)

    ### Train the core network (again)
    if args.train_trunk and (args.retrain_trunk or not args.gate_on_trunk):
        if train_loader_spec is not None and len(train_loader_spec) > 0:
            print("training trunk final")
            dTNet.trunk_net.restore_freeze(trunk_trainability_state)
            layers = get_layers(args.train_mode, dTNet.trunk_cnet, n_attack_layers=args.n_attack_layers,
                                min_layer=last_gate_layer, protected_layers=args.protected_layers, base_layers=False)
            epoch = train_episode(dTNet.trunk_cnet, dTNet, device, args, lr, epoch, n_epochs, train_loader_spec,
                                  test_loader_spec, -1, layers, stats=stats, eps_init=None if (
                            (args.retrain_trunk and args.gate_on_trunk) or (
                                args.load_model is None and args.gate_type == "net")) else 0)
        else:
            raise RuntimeWarning("Core-Network not reached by any training samples. Training skipped.")

    return epoch


def train_episode(cnets, dTNet, device, args, lr, epoch, n_epochs, train_loader, test_loader, episode_idx, layers,
                  stats=None, eps_init=0, balanced_loss=False, net_weights=[1]):
    if not isinstance(cnets,list):
        cnets = [cnets]

    for cnet in cnets: cnet.train()

    net = cnets[0].net
    relaxed_net = cnets[0].relaxed_net

    relu_ids = relaxed_net.get_relu_ids()
    eps = eps_init

    if "COLT" in args.train_mode:
        relu_stable = args.relu_stable
    elif "adv" in args.train_mode:
        relu_stable = None
        args.mix = False
    elif "natural" in args.train_mode:
        relu_stable = None
        args.nat_factor = 1
        args.mix = False
    elif "diffAI" in args.train_mode:
        relu_stable = None
    else:
        raise RuntimeError(f"Unknown train mode {args.train_mode:}")

    print('Saving model to:', args.model_dir)
    print('Training layers: ', layers)

    for j in range(len(layers) - 1):
        opt, lr_scheduler = get_opt(net, args.opt, lr, args.lr_step, args.lr_factor, args.n_epochs, train_loader,
                                    args.lr_sched)
        curr_layer_idx = layers[j + 1]

        eps_old = eps
        eps = get_scaled_eps(args, layers, relu_ids, curr_layer_idx, j)
        if eps_old is None: eps_old = eps
        kappa_sched = Scheduler(0 if args.mix else 1, 1, args.train_batch * len(train_loader) * args.mix_epochs,
                                0 if not args.anneal else args.train_batch * len(train_loader)*args.anneal_warmup)
        beta_sched = Scheduler(args.beta_start if args.mix else args.beta_end, args.beta_end,
                                args.train_batch * len(train_loader) * args.mix_epochs, 0)
        eps_sched = Scheduler(eps_old if args.anneal else eps, eps, args.train_batch * len(train_loader) * args.anneal_epochs,
                              args.train_batch * len(train_loader)*args.anneal_warmup, power=args.anneal_pow)

        layer_dir = '{}/{}/{}'.format(args.model_dir, episode_idx, curr_layer_idx)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)

        print('\nnew train phase: eps={:.5f}, lr={:.2e}, curr_layer={}\n'.format(eps, lr, curr_layer_idx))

        if balanced_loss:
            assert cnets[0].lossFn_test is None, "Unexpected lossFn"
            data_balance = np.array(train_loader.dataset.targets).astype(float).mean()
            balance_factor = (1 - data_balance) / (data_balance + 1e-3)
            cnets[0].update_loss_fn(balance_factor, device)

        for curr_epoch in range(n_epochs):
            if balanced_loss and args.sliding_loss_balance is not None and j == 0:
                # if sliding loss balance is acitve, anneal loss balance from fully balanced to partially balanced
                assert 0 <= args.sliding_loss_balance <= 1
                balance_factor_initial = (1-data_balance)/(data_balance+1e-3)
                scaling_factor_balance = 1-max(min((curr_epoch-0.1*n_epochs)/(n_epochs*0.7), args.sliding_loss_balance), 0)
                balance_factor = scaling_factor_balance * (balance_factor_initial-1) + 1
                cnets[0].update_loss_fn(balance_factor, device)

            train(device, epoch, args, j + 1, layers, cnets, eps_sched, kappa_sched, opt, train_loader,
                  lr_scheduler, relu_ids, stats, relu_stable,
                  relu_stable_protected=args.relu_stable_protected, net_weights=net_weights, beta_sched=beta_sched)

            if isinstance(lr_scheduler, optim.lr_scheduler.StepLR) and curr_epoch >= args.mix_epochs:
                lr_scheduler.step()

            if (epoch + 1) % args.test_freq == 0 or (epoch + 1) % n_epochs == 0:
                torch.save(dTNet.state_dict(), os.path.join(layer_dir, 'net_%d.pt' % (epoch + 1)))
                torch.save(dTNet.state_dict(), os.path.join(layer_dir, 'opt_%d.pt' % (epoch + 1)))
                test(device, args, cnets[0], test_loader if args.test_set == "test" else train_loader,
                     [curr_layer_idx], stats=stats)

            stats.update_tb(epoch)
            epoch += 1
        relu_stable = None if relu_stable is None else relu_stable * args.relu_stable_layer_dec
        lr = lr * args.lr_layer_dec
    net.freeze(len(net.blocks)-1)
    return epoch


def get_gated_data_loaders(args, device, cnet, train_loader, test_loader, stats, mode="gate", exact=None, threshold=0):
    if exact is None:
        exact = True if args.train_domain[0] == "box" else False

    gate_train_loader, class_list = get_gated_data(args, device, cnet, train_loader, stats, mode, exact, is_train=True, threshold=threshold)
    if mode.startswith("gate_c_mat_"):
        mode = "gate_%s" % "_".join([str(x) for x in class_list])
    gate_test_loader, _ = get_gated_data(args, device, cnet, test_loader, stats, mode, exact, is_train=False, threshold=threshold)

    return gate_train_loader, gate_test_loader


def get_gated_data(args, device, cnet, data_loader, stats, mode="gate", exact=False, is_train=True, threshold=0):
    if mode in ["gate_init","gate_init_adv"]:
        cert_mode = "acc"
    elif mode == "branch":
        cert_mode = 0
    elif mode == "trunk":
        cert_mode = 1
    else:
        assert False, "cert_mode: %s not recognized" % (mode)

    class_list = None

    data_set_tmp = MyVisionDataset.from_idx(data_loader.dataset, np.ones_like(data_loader.dataset.targets).astype(bool))
    data_loader_tmp = torch.utils.data.DataLoader(data_set_tmp, batch_size=data_loader.batch_size,
                                                  shuffle=False, num_workers=data_loader.num_workers, pin_memory=True, drop_last=False)
    if mode == "branch" and args.gate_type == "entropy":
        return data_loader, class_list
    else:
        threshold_cert, threshold_adv, y, nat_y, adv_y, sample_weights = diffAI_cert(device, args, cnet, data_loader_tmp, stats=stats,
                                                                     log_ind=True,
                                                                     cert_mode=cert_mode, exact=exact, ind_class=True,
                                                                     domains=args.cert_domain[0:1], print_table=False,
                                                                     threshold_min=threshold)
        if "adv" in mode:
            threshold_cert = threshold_adv
        else:
            threshold_cert = threshold_cert[args.cert_domain[0]]

    if mode in ["branch", "trunk"]:
        # get not certified
        gate_target = (threshold_cert < threshold).cpu().detach()
    elif "gate_init" in mode:
        # get not certified
        if args.gate_on_trunk:
            gate_target = (threshold_cert < threshold).cpu().detach()
        else:
            gate_target = (threshold_cert > threshold).cpu().detach()
    else:
        raise RuntimeError(f"gate iteration mode {mode:} not recognized")

    if "gate" in mode:
        data_set = MyVisionDataset.from_idx_and_targets(data_loader.dataset, torch.ones_like(gate_target).numpy(),
                                                        gate_target.float(), ["gate_easy_cert", "gate_difficult_cert"])
    else:
        if args.weighted_training is not None:
            data_set = MyVisionDataset.from_idx(data_loader.dataset, torch.ones_like(gate_target).numpy())
            weights = torch.tensor(data_set.sample_weights) * (1-(1-args.weighted_training)*(~gate_target.bool()))
            data_set.set_weights(weights)
        else:
            data_set = MyVisionDataset.from_idx(data_loader.dataset, gate_target.bool().numpy())

    if len(data_set) == 0:
        return None, class_list
    else:
        return torch.utils.data.DataLoader(data_set, batch_size=args.train_batch if is_train else args.test_batch,
                                       shuffle=~args.debug if is_train else False, num_workers=data_loader.num_workers,
                                       pin_memory=True, drop_last=False), class_list


def ai_cert_sample(dTNet, inputs, target, branch_p, domain, eps, break_on_failure, cert_trunk=True):
    dTNet.eval()
    ver_corr = torch.ones_like(target).byte()
    ver_not_trunk = False
    gate_threshold_s = {}
    n_class = dTNet.gate_nets[dTNet.exit_ids[1]].blocks[-1].out_features

    for k, exit_idx in enumerate(dTNet.exit_ids[1:]):
        if branch_p[k+1] == 0:
            ver_not_trunk = False
            ver_not_branch = True
            continue

        # try:
        ver_not_branch, non_selection_threshold, _ = dTNet.gate_cnets[exit_idx].get_abs_loss(inputs,
                                                                       torch.zeros_like(target).int(), eps, domain,
                                                                       dTNet.threshold[exit_idx],beta=1)
        ver_not_trunk, selection_threshold, _ = dTNet.gate_cnets[exit_idx].get_abs_loss(inputs,
                                                                       torch.ones_like(target).int(), eps, domain,
                                                                       dTNet.threshold[exit_idx],beta=1)
        # gate_threshold_s[exit_idx] = torch.stack((-non_selection_threshold,selection_threshold),dim=0).gather(dim=0, index=targets_abs.view(1, -1, 1)).squeeze(0)

        try:
            pass
        except:
            print("Verification of gate failed")
            # ver = torch.zeros_like(target).byte()
            ver_not_branch = torch.zeros_like(target).byte()
            # gate_threshold_s[exit_idx] = -np.inf
        # ver_not_trunk = ver - ver_not_branch

        if ver_not_branch:
            # Sample can not reach branch
            branch_p[k + 1] = 0
        else:
            # Sample can reach branch
            if branch_p[k+1] == 2:
                # Already certified
                ver_corr_branch = torch.ones_like(target).byte()
                # ver = torch.ones_like(target).byte()
            elif branch_p[k+1] == -1:
                # Already certified as incorrect
                ver_corr_branch = torch.zeros_like(target).byte()
                # ver = torch.ones_like(target).byte()
            else:
                branch_p[k + 1] = 1

                ver_corr_branch, _, _ = dTNet.branch_cnets[exit_idx].get_abs_loss(inputs, target, eps, domain,
                                                                                    dTNet.branch_cnets[exit_idx].threshold_min,beta=1)
                ver_corr_branch = ver_corr_branch.byte()
                # ver_branch = ver_branch.byte()


                if ver_corr_branch:
                    branch_p[k + 1] = 2
                # elif ver_branch:
                #     branch_p[k + 1] = -1
            # ver corr if all reachable branches are correct
            ver_corr = ver_corr_branch & ver_corr

        if ver_not_trunk:
            # Will definitely branch => all later branches cannot be reached
            branch_p[0] = 0
            if k+2 < len(branch_p):
                for i, reachability in enumerate(branch_p[k+2:]):
                    branch_p[k + 2 + i] = 0
            break

        if not ver_corr and break_on_failure:
            # assume that all branches can be reached if the opposite was not certified
            if not ver_not_trunk:
                for i, reachability in enumerate(branch_p[k+1:]):
                    if reachability == 0:
                        branch_p[k + 1 + i] = 1
            break

    if not ver_not_trunk and not branch_p[0] == 0:
        if ver_corr or not break_on_failure:
            if cert_trunk:
                try:
                    ver_corr_trunk, threshold_n, _ = dTNet.trunk_cnet.get_abs_loss(inputs, target, eps, domain,
                                                                                    dTNet.trunk_cnet.threshold_min,kappa=1)
                    if ver_corr_trunk:
                        branch_p[0] = 2
                    # elif ver_trunk:
                    #     branch_p[0] = -1
                    ver_corr = ver_corr_trunk & ver_corr
                except:
                    warn("Certification of trunk failed critically.")
                    ver_corr[:] = False
            else:
                ver_corr[:] = False
    return branch_p, ver_corr, gate_threshold_s


def cert_deepTrunk_net(dTNet, args, device, data_loader, stats, log_ind=True, break_on_failure=False, epoch=None, domains=None):
    if domains is None: domains = args.cert_domain
    dTNet.eval()
    n_exits = dTNet.n_branches
    tot_branches_p = [0 for _ in range(n_exits+1)]
    tot_select_n = torch.tensor([0. for _ in range(n_exits+1)]).to(device)
    tot_select_hist = torch.tensor([0. for _ in range(n_exits+1)]).to(device)
    tot_ver_corr = {domain: 0 for domain in domains}
    tot_corr = 0
    tot_adv_corr = 0
    tot_samples = 0
    img_id = 0
    gate_threshold_aggregate = []

    adv_attack_test = AdvAttack(eps=args.test_eps, n_steps=args.test_att_n_steps,
                                step_size=args.test_att_step_size, adv_type="pgd")

    if log_ind:
        data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size,
                                    shuffle=False, num_workers=data_loader.num_workers, pin_memory=True,
                                    drop_last=False)
        csv_log = open(os.path.join(args.model_dir, "cert_log.csv"), mode='w')
        log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cert_domain_list = ["ver_ok_%s" % domain for domain in domains]
        log_writer.writerow(["img_id", "label", "nat_ok", "pgd_ok"] + cert_domain_list + ["nat_branch", "pgd_branch"]
                            + ["branch_%d_p" % i for i in dTNet.exit_ids]
                            + ["branch_%d_adv_p" % i for i in dTNet.exit_ids])
                            # + ["gate_%d_%s_threshold" % (i,d) for d in domains for i in dTNet.exit_ids[1:]])

    pbar = tqdm(data_loader, dynamic_ncols=True)

    for it, (inputs, targets) in enumerate(pbar):
        if args.debug and it == 50:
            pbar.close()
            break

        inputs, targets = inputs.to(device), targets.to(device)
        tot_samples += inputs.size()[0]
        nat_out, gate_out, exit_mask, nat_branches = dTNet(inputs)
        if dTNet.trunk_net is None:
            exit_mask[-2] = exit_mask[-2].__or__(exit_mask[-1])
            exit_mask = exit_mask[:-1]
        nat_ok = targets.eq(dTNet.evalFn(nat_out[torch.stack(exit_mask, dim=1)])).detach()
        _, adv_ok, _, _, adv_branches = dTNet.get_adv_loss(inputs, targets, adv_attack_test)
        select_mask_adv, min_gate_out, max_gate_out = dTNet.get_adv_loss(inputs, targets, adv_attack_test, mode="reachability")

        tot_select_hist += select_mask_adv.sum(dim=0).detach()
        tot_select_n += (torch.arange(1, n_exits + 2, device=device).unsqueeze(dim=0) == select_mask_adv.sum(
            dim=1).unsqueeze(dim=1)).sum(dim=0).detach()

        for n in range(inputs.shape[0]):
            img_id += 1
            target = targets[n:n + 1]
            input_img = inputs[n:n + 1]
            branch_p_aggregate = np.array([1 for _ in range(dTNet.n_branches + 1)])
            cert_list = []
            # gate_threshold_list = []

            with torch.no_grad():
                for domain in domains:
                    branch_p = [1 for _ in range(dTNet.n_branches + 1)] # 0 => cant reach, 1 => reachable, 2 => cert, -1 => inccorect class cert

                    branch_p, ver_corr, gate_thresholds = ai_cert_sample(dTNet, input_img, target, branch_p, domain, adv_attack_test.eps, break_on_failure,
                                                        False if dTNet.trunk_net is None else args.cert_trunk)

                    tot_ver_corr[domain] += ver_corr.int().item()
                    cert_list += [ver_corr.int().item()]
                    # gate_threshold_list += [gate_thresholds[x].item() for x in dTNet.exit_ids[1:]]
                    branch_p_idx = branch_p_aggregate == 1
                    branch_p_aggregate[branch_p_idx] = np.array(branch_p)[branch_p_idx]

                tot_branches_p = [int(x1 != 0) + x2 for x1,x2 in zip(branch_p_aggregate, tot_branches_p)]
                # gate_threshold_aggregate += [torch.Tensor(gate_threshold_list)]
            if log_ind:
                log_writer.writerow([img_id, target.item(), nat_ok[n].int().item(), adv_ok[n].int().item()]
                                    + cert_list + [ nat_branches[n].int().item(), adv_branches[n].int().item()]
                                    + list(branch_p_aggregate)
                                    + [select_mask_adv[n,-1].int().item()]
                                    + list(select_mask_adv[n,0:-1].int().cpu().numpy()))
                                    # + gate_threshold_list)

        tot_corr += nat_ok.int().sum().item()
        tot_adv_corr += adv_ok.int().sum().item()


        msg_nat = '%.4f' % (tot_corr / tot_samples)
        msg_adv = '%.4f' % (tot_adv_corr / tot_samples)
        msg_reachable = ";  ".join(['%d: %.4f/%.4f' % (idx, tot_select_hist[i-1%(n_exits+1)].item()/tot_samples, x/tot_samples) for
                                    i, (idx, x) in enumerate(zip(dTNet.exit_ids, tot_branches_p))])
        msg_ver_corr = ", ".join(['%s: %.4f' % (domain, tot_ver_corr[domain]/tot_samples) for domain in domains])
        pbar.set_description('[C] tot_nat= %s, tot_adv= %s, tot_reachable(adv/cert)= {%s}, tot_ver_corr= {%s}' %
                             (msg_nat, msg_adv, msg_reachable, msg_ver_corr))

    if log_ind:
        log_writer.writerow(
            ["total", "", tot_corr / tot_samples, tot_adv_corr / tot_samples]
             + [tot_ver_corr[domain] / tot_samples for domain in domains]
             + ["", ""]+[x / tot_samples for x in tot_branches_p]
             + [tot_select_hist[i-1%(n_exits+1)].item()/tot_samples for i in range(n_exits+1)])
        csv_log.close()

    if stats is not None:
        for domain in tot_ver_corr.keys():
            stats.report('cert/acc_ver/%s' % domain, tot_ver_corr[domain]/tot_samples)
        stats.report('cert/acc_nat', tot_corr / tot_samples)
        stats.report('cert/acc_adv', tot_adv_corr / tot_samples)
        for i, exit_idx in enumerate(dTNet.exit_ids):
            stats.report('cert/exit_%d_reached' % exit_idx, tot_branches_p[i]/ tot_samples)
        stats.update_tb(epoch)


def test_deepTrunk_net(dTNet, args, device, data_loader, stats, log_ind=True, epoch=None):
    dTNet.eval()
    n_exits = dTNet.n_branches
    tot_branches_p = [0 for _ in range(n_exits+1)]
    tot_select_n = torch.tensor([0. for _ in range(n_exits+1)]).to(device)
    tot_select_hist = torch.tensor([0. for _ in range(n_exits+1)]).to(device)
    tot_corr = 0
    tot_adv_corr = 0
    tot_samples = 0
    img_id = 0

    adv_attack_test = AdvAttack(eps=args.test_eps, n_steps=args.test_att_n_steps,
                                step_size=args.test_att_step_size, adv_type="pgd")

    if log_ind:
        data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size,
                                    shuffle=False, num_workers=data_loader.num_workers, pin_memory=True,
                                    drop_last=False)
        csv_log = open(os.path.join(args.model_dir, "test_log.csv"), mode='w')
        log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow(["img_id", "label", "nat_ok", "pgd_ok", "nat_branch", "pgd_branch"]
                            + ["branch_%d_adv_p" % i for i in dTNet.exit_ids]
                            + ["nat_gate_%d_out_p" % i for i in dTNet.exit_ids[1:]]
                            + ["min_gate_%d_out_p" % i for i in dTNet.exit_ids[1:]]
                            + ["max_gate_%d_out_p" % i for i in dTNet.exit_ids[1:]])

    pbar = tqdm(data_loader, dynamic_ncols=True)

    for it, (inputs, targets) in enumerate(pbar):
        if args.debug and it == 50:
            pbar.close()
            break

        inputs, targets = inputs.to(device), targets.to(device)
        tot_samples += inputs.size()[0]
        nat_out, nat_gate_out, exit_mask, nat_branches = dTNet(inputs)
        if dTNet.trunk_net is None:
            exit_mask[-2] = exit_mask[-2].__or__(exit_mask[-1])
            exit_mask = exit_mask[:-1]
        nat_ok = targets.eq(dTNet.evalFn(nat_out[torch.stack(exit_mask, dim=1)])).detach()
        _, adv_ok, _, _, adv_branches = dTNet.get_adv_loss(inputs, targets, adv_attack_test)
        select_mask_adv, select_gate_out_min, select_gate_out_max = dTNet.get_adv_loss(inputs, targets, adv_attack_test, mode="reachability")

        tot_select_hist += select_mask_adv.sum(dim=0).detach()
        tot_select_n += (torch.arange(1, n_exits + 2, device=device).unsqueeze(dim=0) == select_mask_adv.sum(
            dim=1).unsqueeze(dim=1)).sum(dim=0).detach()

        if log_ind:
            for n in range(len(targets)):
                img_id += 1
                log_writer.writerow([img_id, targets[n].int().item(), nat_ok[n].int().item(), adv_ok[n].int().item()]
                                + [ nat_branches[n].int().item(), adv_branches[n].int().item()]
                                + [select_mask_adv[n, -1].int().item()]
                                + list(select_mask_adv[n, 0:-1].int().cpu().numpy())
                                + list(nat_gate_out[n].float().cpu().detach().view(-1).numpy())
                                + list(select_gate_out_min[n].float().cpu().detach().view(-1).numpy())
                                + list(select_gate_out_max[n].float().cpu().detach().view(-1).numpy()))

        tot_corr += nat_ok.int().sum().item()
        tot_adv_corr += adv_ok.int().sum().item()

        msg_nat = '%.4f' % (tot_corr / tot_samples)
        msg_adv = '%.4f' % (tot_adv_corr / tot_samples)
        msg_reachable = ";  ".join(['%d: %.4f' % (idx, tot_select_hist[i-1%(n_exits+1)].item()/tot_samples) for
                                    i, idx in enumerate(dTNet.exit_ids)])
        pbar.set_description('[C] tot_nat= %s, tot_adv= %s, tot_reachable(adv)= {%s}' %
                             (msg_nat, msg_adv, msg_reachable))

    if log_ind:
        log_writer.writerow(
            ["total", "", tot_corr / tot_samples, tot_adv_corr / tot_samples]
             + ["", ""]
             + [tot_select_hist[i-1%(n_exits+1)].item()/tot_samples for i in range(n_exits+1)]
             + [""]*2*len(dTNet.exit_ids[1:]))
        csv_log.close()

    if stats is not None:
        stats.report('cert/acc_nat', tot_corr / tot_samples)
        stats.report('cert/acc_adv', tot_adv_corr / tot_samples)
        for i, exit_idx in enumerate(dTNet.exit_ids):
            stats.report('cert/exit_%d_reached' % exit_idx, tot_branches_p[i]/ tot_samples)
        stats.update_tb(epoch)


def run(args=None):
    device = 'cuda' if torch.cuda.is_available() and (not args.no_cuda) else 'cpu'
    num_train, train_loader, test_loader, input_size, input_channel, n_class = get_loaders(args)

    lossFn = nn.CrossEntropyLoss(reduction='none')
    def evalFn(x): return torch.max(x, dim=1)[1]

    ## initialize SpecNet
    dTNet = MyDeepTrunkNet.get_deepTrunk_net(args, device, lossFn, evalFn, input_size, input_channel, n_class)

    ## setup logging and checkpointing
    timestamp = int(time.time())
    model_signature = '%s/%s/%d/%s_%.5f/%d' % (args.dataset, args.exp_name, args.exp_id, args.net, args.train_eps, timestamp)
    model_dir = args.root_dir + 'models_new/%s' % (model_signature)
    args.model_dir = model_dir


    print("Saving model to: %s" % model_dir)
    count_vars(args, dTNet)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    tb_writer = SummaryWriter(model_dir)
    stats = Statistics(len(train_loader), tb_writer, model_dir)
    args_file = os.path.join(model_dir, 'args.json')
    with open(args_file, 'w') as fou:
        json.dump(vars(args), fou, indent=4)
    write_config(args, os.path.join(model_dir, 'run_config.txt'))


    ## main part depending on training mode
    if 'train' in args.train_mode:
        epoch = train_deepTrunk(dTNet, args, device, stats, train_loader, test_loader)
        if args.cert:
            with torch.no_grad():
                cert_deepTrunk_net(dTNet, args, device, test_loader if args.test_set == "test" else train_loader,
                                   stats, log_ind=True, break_on_failure=False, epoch=epoch)
    elif args.train_mode == 'test':
        with torch.no_grad():
            test_deepTrunk_net(dTNet, args, device, test_loader if args.test_set == "test" else train_loader, stats,
                               log_ind=True)
    elif args.train_mode == "cert":
        with torch.no_grad():
            cert_deepTrunk_net(dTNet, args, device, test_loader if args.test_set == "test" else train_loader, stats,
                               log_ind=True, break_on_failure=False)
    else:
        assert False, 'Unknown mode: {}!'.format(args.train_mode)

    exit(0)


def main():
    args = get_args()
    run(args=args)


if __name__ == '__main__':
    main()
