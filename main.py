import time
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from args_factory import get_args
from loaders import get_loaders
from utils import (get_net, Scheduler, Statistics, count_vars, write_config, get_scaled_eps, get_layers)
from relaxed_networks import RelaxedNetwork, CombinedNetwork
from networks import UpscaleNet
from trainer import train, test, get_opt, diffAI_cert
import random


seed = 100

torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.set_printoptions(precision=10)
np.random.seed(seed)
random.seed(seed)


def run(args=None):
    device = 'cuda' if torch.cuda.is_available() and (not args.no_cuda) else 'cpu'
    num_train, train_loader, test_loader, input_size, input_channel, n_class = get_loaders(args)

    lossFn = nn.CrossEntropyLoss(reduction='none')
    evalFn = lambda x: torch.max(x, dim=1)[1]

    net = get_net(device, args.dataset, args.net, input_size, input_channel, n_class, load_model=args.load_model,
                  net_dim=args.cert_net_dim)#, feature_extract=args.core_feature_extract)

    timestamp = int(time.time())
    model_signature = '%s/%s/%d/%s_%.5f/%d' % (args.dataset, args.exp_name, args.exp_id, args.net, args.train_eps, timestamp)
    model_dir = args.root_dir + 'models_new/%s' % (model_signature)
    args.model_dir = model_dir
    count_vars(args, net)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if isinstance(net, UpscaleNet):
        relaxed_net = None
        relu_ids = None
    else:
        relaxed_net = RelaxedNetwork(net.blocks, args.n_rand_proj).to(device)
        relu_ids = relaxed_net.get_relu_ids()

    if "nat" in args.train_mode:
        cnet = CombinedNetwork(net, relaxed_net, lossFn=lossFn, evalFn=evalFn, device=device, no_r_net=True).to(device)
    else:
        dummy_input = torch.rand((1,)+net.dims[0],device=device, dtype=torch.float32)
        cnet = CombinedNetwork(net, relaxed_net, lossFn=lossFn, evalFn=evalFn, device=device, dummy_input=dummy_input).to(device)


    n_epochs, test_nat_loss, test_nat_acc, test_adv_loss, test_adv_acc = args.n_epochs, None, None, None, None

    if 'train' in args.train_mode:
        tb_writer = SummaryWriter(model_dir)
        stats = Statistics(len(train_loader), tb_writer, model_dir)
        args_file = os.path.join(model_dir, 'args.json')
        with open(args_file, 'w') as fou:
            json.dump(vars(args), fou, indent=4)
        write_config(args, os.path.join(model_dir, 'run_config.txt'))

        eps = 0
        epoch = 0
        lr = args.lr
        n_epochs = args.n_epochs

        if "COLT" in args.train_mode:
            relu_stable = args.relu_stable
            # if args.layers is None:
            #     args.layers = [-2, -1] + relu_ids
            layers = get_layers(args.train_mode, cnet, n_attack_layers=args.n_attack_layers, protected_layers=args.protected_layers)
        elif "adv" in args.train_mode:
            relu_stable = None
            layers = [-1, -1]
            args.mix = False
        elif "natural" in args.train_mode:
            relu_stable = None
            layers = [-2, -2]
            args.nat_factor = 1
            args.mix = False
        elif "diffAI" in args.train_mode:
            relu_stable = None
            layers = [-2, -2]
        else:
            assert False, "Unknown train mode %s" % args.train_mode

        print('Saving model to:', model_dir)
        print('Training layers: ', layers)

        for j in range(len(layers)-1):
            opt, lr_scheduler = get_opt(cnet.net, args.opt, lr, args.lr_step, args.lr_factor, args.n_epochs,
                                        train_loader, args.lr_sched, fixup="fixup" in args.net)

            curr_layer_idx = layers[j+1]
            eps_old = eps
            eps = get_scaled_eps(args, layers, relu_ids, curr_layer_idx, j)

            kappa_sched = Scheduler(0.0 if args.mix else 1.0, 1.0, num_train * args.mix_epochs, 0)
            beta_sched = Scheduler(args.beta_start if args.mix else args.beta_end, args.beta_end,
                                   args.train_batch * len(train_loader) * args.mix_epochs, 0)
            eps_sched = Scheduler(eps_old if args.anneal else eps, eps, num_train * args.anneal_epochs, 0)

            layer_dir = '{}/{}'.format(model_dir, curr_layer_idx)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)

            print('\nnew train phase: eps={:.5f}, lr={:.2e}, curr_layer={}\n'.format(eps, lr, curr_layer_idx))

            for curr_epoch in range(n_epochs):
                train(device, epoch, args, j+1, layers, cnet, eps_sched, kappa_sched, opt, train_loader,
                      lr_scheduler, relu_ids, stats, relu_stable,
                      relu_stable_protected=args.relu_stable_protected, beta_sched=beta_sched)

                if isinstance(lr_scheduler, optim.lr_scheduler.StepLR) and curr_epoch >= args.mix_epochs:
                    lr_scheduler.step()

                if (epoch + 1) % args.test_freq == 0:
                    with torch.no_grad():
                        test_nat_loss, test_nat_acc, test_adv_loss, test_adv_acc = test(device, args, cnet,
                                                            test_loader if args.test_set == "test" else train_loader,
                                                            [curr_layer_idx], stats=stats, log_ind=(epoch + 1) % n_epochs == 0)

                if (epoch + 1) % args.test_freq == 0 or (epoch + 1) % n_epochs == 0:
                    torch.save(net.state_dict(), os.path.join(layer_dir, 'net_%d.pt' % (epoch + 1)))
                    torch.save(opt.state_dict(), os.path.join(layer_dir, 'opt_%d.pt' % (epoch + 1)))

                stats.update_tb(epoch)
                epoch += 1
            relu_stable = None if relu_stable is None else relu_stable * args.relu_stable_layer_dec
            lr = lr * args.lr_layer_dec
        if args.cert:
            with torch.no_grad():
                diffAI_cert(device, args, cnet, test_loader if args.test_set == "test" else train_loader, stats=stats,
                            log_ind=True, epoch=epoch, domains=args.cert_domain)
    elif args.train_mode == 'print':
        print('printing network to:', args.out_net_file)
        dummy_input = torch.randn(1, input_channel, input_size, input_size, device='cuda')
        net.skip_norm = True
        torch.onnx.export(net, dummy_input, args.out_net_file, verbose=True)
    elif args.train_mode == 'test':
        with torch.no_grad():
            test(device, args, cnet, test_loader if args.test_set == "test" else train_loader, [-1], log_ind=True)
    elif args.train_mode == "cert":
        tb_writer = SummaryWriter(model_dir)
        stats = Statistics(len(train_loader), tb_writer, model_dir)
        args_file = os.path.join(model_dir, 'args.json')
        with open(args_file, 'w') as fou:
            json.dump(vars(args), fou, indent=4)
        write_config(args, os.path.join(model_dir, 'run_config.txt'))
        print('Saving results to:', model_dir)
        with torch.no_grad():
            diffAI_cert(device, args, cnet, test_loader if args.test_set == "test" else train_loader, stats=stats,
                        log_ind=True, domains=args.cert_domain)
        exit(0)
    else:
        assert False, 'Unknown mode: {}!'.format(args.train_mode)

    return test_nat_loss, test_nat_acc, test_adv_loss, test_adv_acc


def main():
    args = get_args()
    run(args=args)


if __name__ == '__main__':
    main()
