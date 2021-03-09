import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import pickle
from scipy.stats import norm

import re
import json
import os
from layers import Linear, Conv2d
from networks import FFNN, ConvMedBig,  MyResnet, myNet, EfficientNet
from itertools import combinations
from PIL import Image
from warnings import warn


def get_network(device, dataset, net_name, input_size, input_channel, n_class, net_config=None, net_dim=None):
    if net_name.startswith('ffnn_'):
        tokens = net_name.split('_')
        sizes = [int(x) for x in tokens[1:]]
        net = FFNN(device, dataset, sizes, n_class, input_size, input_channel, net_dim=net_dim).to(device)
    elif net_name.startswith('convmedbig_'):
        tokens = net_name.split('_')
        obj = ConvMedBig
        assert tokens[0] == 'convmedbig'
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        width3 = int(tokens[4])
        linear_size = int(tokens[5])
        net = obj(device, dataset, n_class, input_size, input_channel, width1, width2, width3, linear_size=linear_size).to(device)
    elif net_name.startswith('resnet'):
        tokens = net_name.split('_')
        net = MyResnet(device, dataset, [1, 2], n_class, input_size, input_channel, block=tokens[1], net_dim=net_dim).to(device)
    elif net_name.startswith('myresnet'):
        tokens = net_name.split('_')
        if "p" in tokens[-1]:
            pool = tokens[-1][1:]
            tokens = tokens[:-1]
            if pool == "g":
                pooling = "global"
            elif pool == "n":
                pooling = None
            else:
                pooling = int(pool)
        else:
            pooling = "global"

        if "w" in tokens[-1]:
            widen = tokens[-1][1:]
            tokens = tokens[:-1]
            widen = int(widen)
        else:
            widen = 1

        n_blocks = list(map(int, tokens[2:]))
        net = MyResnet(device, dataset, n_blocks, n_class, input_size, input_channel, block=tokens[1], in_planes=16,
                       net_dim=net_dim, pooling=pooling, widen_factor=widen).to(device)
    elif net_name.startswith("myNetMax"):
        if net_config is None:
            net_config = parse_net_config(net_name, n_class, input_size, input_channel)
            net_config["max"] = True
            net_config["scale_width"] = False
        net = myNet(device, dataset, net_dim=net_dim, **net_config).to(device)
    elif net_name.startswith("myNet"):
        if net_config is None:
            net_config = parse_net_config(net_name, n_class, input_size, input_channel)
        net = myNet(device, dataset, net_dim=net_dim, **net_config).to(device)
    elif net_name.startswith("efficientnet-"):
        tokens = net_name.split("_")
        pretrained = "pre" in tokens
        adv = "adv" in tokens
        net = EfficientNet(device, dataset, tokens[0], input_size, input_channel, n_class, pretrained, adv)
    else:
        assert False, 'Unknown network!'
    net.determine_dims(torch.randn((2, input_channel, input_size, input_size), dtype=torch.float).to(device))
    return net


def parse_net_config(net_name, n_class, input_size, input_channel):
    tokens = [x.split("_") for x in net_name.split("__")[1:]]
    conv_widths = [0]
    kernel_size = [0]
    strides = [1]
    depth_conv = [0]
    paddings = None
    pool = False
    bn = re.match("myNet-BN.*",net_name) is not None
    if len(tokens) == 1:
        linear_sizes = tokens[0]
    elif len(tokens) == 3:
        conv_widths, kernel_size, linear_sizes = tokens
    elif len(tokens) == 4:
        conv_widths, kernel_size, depth_conv, linear_sizes = tokens
    elif len(tokens) == 5:
        conv_widths, kernel_size, strides, depth_conv, linear_sizes = tokens
    elif len(tokens) == 6:
        conv_widths, kernel_size, strides, paddings, depth_conv, linear_sizes = tokens
    elif len(tokens) == 7:
        conv_widths, kernel_size, strides, paddings, depth_conv, linear_sizes, pool = tokens
        if pool == 1:
            pool = True
    else:
        raise RuntimeError("Cant read net configuration")

    conv_widths = [int(x) for x in conv_widths]
    kernel_size = [int(x) for x in kernel_size]
    strides = [int(x) for x in strides]
    paddings = None if paddings is None else [int(x) for x in paddings]
    if conv_widths[0] == 0:
        conv_widths = []
        kernel_size = []
        strides = []
    linear_sizes = [int(x) for x in linear_sizes]
    depth_conv = int(depth_conv[0]) if int(depth_conv[0]) > 0 else None
    net_config = {"n_class": n_class, "input_size": input_size, "input_channel": input_channel,
                  "conv_widths": conv_widths, "kernel_sizes": kernel_size, "linear_sizes": linear_sizes,
                  "depth_conv": depth_conv, "paddings": paddings, "strides": strides, "pool": pool, "bn": bn}
    return net_config


def get_net(device, dataset, net_name, input_size, input_channel, n_class, load_model=None, net_config=None, balance_factor=1, net_dim=None):
    net = get_network(device, dataset, net_name, input_size, input_channel, n_class, net_config=net_config, net_dim=net_dim).to(device) #, feature_extract=-1).to(device)

    if n_class == 1 and isinstance(net.blocks[-1], Linear) and net.blocks[-1].bias is not None:
        net.blocks[-1].linear.bias.data = torch.tensor(-norm.ppf(balance_factor/(balance_factor+1)),
                                                       dtype=net.blocks[-1].linear.bias.data.dtype).view(net.blocks[-1].linear.bias.data.shape)

    if load_model is not None:
        if "crown-ibp" in load_model or "LiRPA" in load_model:
            net = load_CROWNIBP_net_state(net, load_model)
        else:
            net = load_net_state(net, load_model)

    init_slopes(net, device, trainable=False)
    return net


def load_net_state(net, load_model):
    state_dict_load = torch.load(load_model)
    state_dict_new = net.state_dict()
    try:
        missing_keys, unexpected_keys = net.load_state_dict(state_dict_load, strict=False)
        print("net loaded from %s. %d missing_keys. %d unexpected_keys" % (load_model, len(missing_keys), len(unexpected_keys)))
    except:
        counter = 0
        for k_load, v_load in state_dict_load.items():
            if k_load in state_dict_new and all(v_load.squeeze().shape == np.array(state_dict_new[k_load].squeeze().shape)):
                state_dict_new.update({k_load: v_load.view(state_dict_new[k_load].shape)})
                counter += 1
        net.load_state_dict(state_dict_new, strict=False)
        print("%d/%d parameters loaded from from %s" % (counter, len(state_dict_new), load_model))

    return net


def match_keys(new_dict, loaded_dict):
    new_dict_keys = list(new_dict.keys())
    new_dict_type = np.array([("bn." if ".bn" in x else "")+re.match(".*\.([a-z,_]+)$", x).group(1) for x in new_dict_keys])
    new_dict_shape = [tuple(x.size()) if len(x.size())>0 else (1,) for x in new_dict.values()]
    unloaded = np.ones(len(new_dict_keys), dtype=int)

    loaded_dict_type = [("bn." if "bn" in x else "")+re.match(".*\.([a-z,_]+)$", x).group(1) for x in loaded_dict.keys()]
    if all([not "bn" in x for x in loaded_dict_type]) and any([not "bn" in x for x in new_dict_type]):
        bn_ids = [re.match("(.*)\.(running_mean)$", x).group(1) for x in loaded_dict.keys() if "running_mean" in x]
        loaded_dict_type = ["bn."+ x if re.match("(.*)\.[a-z,_]+$", y).group(1) in bn_ids else x for x,y in zip(loaded_dict_type,loaded_dict.keys())]

    for i, (k,v) in enumerate(loaded_dict.items()):
        matched_key_ids = (unloaded
                           *(new_dict_type == loaded_dict_type[i])
                           *(np.array([x == (tuple(v.size()) if len(v.size())>0 else (1,)) for x in new_dict_shape]))
                           ).nonzero()[0]
        if len(matched_key_ids)==0:
            warn(f"Model only partially loaded. Failed at [{i+1:d}/{len(loaded_dict_type):d}]")
            break
        matched_key_idx = matched_key_ids[0] if isinstance(matched_key_ids, np.ndarray) else int(matched_key_ids)
        matched_key = new_dict_keys[matched_key_idx]
        key_match = re.match("(.*[0-9]+\.)([a-z]*\.){0,1}([bn.]{0,1}[a-z,_]+)$", matched_key)
        matched_keys = [x for x in new_dict_keys if x.startswith(key_match.group(1)) and x.endswith(key_match.group(3))]
        for j, x in enumerate(new_dict_keys):
            if x in matched_keys and j in matched_key_ids:
                unloaded[j] = 0
                new_dict.update({x: v})

    return new_dict


def load_CROWNIBP_net_state(net, load_model):
    checkpoint = torch.load(load_model)
    state_dict_new = net.state_dict()
    if isinstance(checkpoint["state_dict"], list):
        checkpoint["state_dict"] = checkpoint["state_dict"][0]
    state_dict_load = match_keys(state_dict_new, checkpoint["state_dict"])

    try:
        missing_keys, unexpectet_keys = net.load_state_dict(state_dict_load, strict=False)
        assert len([x for x in missing_keys if "weight" in x or "bias" in x]) == 0 and len(unexpectet_keys) == 0
        print("net loaded from %s" % load_model)
    except:
        counter = 0
        for k_load, v_load in state_dict_load.items():
            if k_load in state_dict_new and v_load.shape == state_dict_new[k_load].shape:
                state_dict_new.update({k_load: v_load})
                counter += 1
        net.load_state_dict(state_dict_new, strict=False)
        print("%d/%d parameters loaded from from %s" % (counter, len(state_dict_load), load_model))

    return net


def my_cauchy(*shape):
    return torch.clamp(torch.FloatTensor(*shape).cuda().cauchy_(), -1e7, 1e7)

class Scheduler:
    def __init__(self, start, end, n_steps, warmup, power=1):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.warmup = warmup
        self.curr_steps = 0
        self.power=power

    def advance_time(self, k_steps):
        self.curr_steps += k_steps

    def get(self):
        if self.n_steps == self.warmup:
            return self.end
        if self.curr_steps < self.warmup:
            return self.start
        elif self.curr_steps > self.n_steps:
            return self.end
        inter_factor = (self.curr_steps - self.warmup) / float(self.n_steps - self.warmup)
        inter_factor = np.power(inter_factor, 1/self.power)
        return self.start + (self.end - self.start) * inter_factor


class Statistics:

    def __init__(self, window_size, tb_writer, log_dir=None, post_fix=None):
        self.window_size = window_size
        self.tb_writer = tb_writer
        self.values = {}
        self.steps = 0
        self.log_dir = log_dir
        self.post_fix = "" if post_fix is None else post_fix

    def update_post_fix(self, post_fix=""):
        self.post_fix = post_fix

    def report(self, metric_name, value):
        metric_name = metric_name + self.post_fix
        if metric_name not in self.values:
            self.values[metric_name] = []
        self.values[metric_name] += [value]

    def export_to_json(self, file, epoch=None):
        epoch = self.steps if epoch is None else epoch
        data = {"epoch_%d/%s"%(epoch, k): np.mean(v) for k, v in self.values.items() if len(v)>0}
        data = self.parse_keys(data)
        with open(file, 'a') as f:
            json.dump(data, f, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None,
                      indent="\t", separators=None, default=None, sort_keys=True)

    def report_hist(self, metric_name, values):
        metric_name = metric_name + self.post_fix
        self.tb_writer.add_histogram(metric_name, values.view(-1).detach().cpu().numpy(), self.steps)

    def print_stats(self):
        print('==============')
        for k, v in self.values.items():
            print('%s: %.5lf' % (k, np.mean(v)))

    def get(self, k, no_post_fix=False):
        k = k if no_post_fix else k+self.post_fix
        return np.mean(self.values[k])

    def update_tb(self, epoch=None):
        if self.log_dir is not None:
            self.export_to_json(os.path.join(self.log_dir, "training_log.json"), epoch=epoch)
        for k, v in self.values.items():
            if len(v) == 0: continue
            self.tb_writer.add_scalar(k, np.mean(v), self.steps if epoch is None else epoch)
            self.values[k] = []
        self.steps += 1

    def parse_keys(self, data_old):
        data_new = {}
        for k, v in data_old.items():
            data_new = self.add_to_leveled_dict(data_new, k, v)
        return data_new

    def add_to_leveled_dict(self, data_dict, label, data):
        if "/" not in label:
            data_dict[label] = data
            return data_dict
        labels = label.split("/")
        if labels[0] in data_dict:
            data_dict[labels[0]] = self.add_to_leveled_dict(data_dict[labels[0]],"/".join(labels[1:]),data)
        else:
            data_dict[labels[0]] = self.add_to_leveled_dict({},"/".join(labels[1:]),data)
        return data_dict


def init_slopes(net, device, trainable=False):
    for param_name, param_value in net.named_parameters():
        if 'deepz' in param_name:
            param_value.data = -torch.ones(param_value.size()).to(device)
            param_value.requires_grad_(trainable)


def count_vars(args, net):
    var_count = 0
    var_count_t = 0
    var_count_relu = 0

    for p_name, params in net.named_parameters():
        if "weight" in p_name or "bias" in p_name:
            var_count += int(params.numel())
            var_count_t += int(params.numel() * params.requires_grad)
        elif "deepz_lambda" in p_name:
            var_count_relu += int(params.numel())

    args.var_count = var_count
    args.var_count_t = var_count_t
    args.var_count_relu = var_count_relu

    print('Number of parameters: ', var_count)


def write_config(args, file_path):
    f=open(file_path,'w+')
    for param in [param for param in dir(args) if not param[0] == "_"]:
        f.write("{:<30} {}\n".format(param + ":", (str)(getattr(args, param))))
    f.close()


class AdvAttack:

    def __init__(self, eps=2./255, n_steps=1, step_size=1.25, adv_type="pgd"):
        self.eps = eps
        self.n_steps = n_steps
        self.step_size = step_size
        self.adv_type = adv_type

    def update_params(self, eps=None, n_steps=None, step_size=None, adv_type=None):
        self.eps = self.eps if eps is None else eps
        self.n_steps = self.n_steps if n_steps is None else n_steps
        self.step_size = self.step_size if step_size is None else step_size
        self.adv_type = self.adv_type if adv_type is None else adv_type

    def get_params(self):
        return self.eps, self.n_steps, self.step_size, self.adv_type


def get_lp_loss(blocks, p=1, input_size=1, scale_conv=True):
    lp_loss = 0
    N = input_size

    for block in blocks:
        if isinstance(block,Conv2d):
            conv = block.conv
            N = max(np.floor((N + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) /
                             conv.stride[0]), 0.0) + 1
            lp_loss += block.weight.abs().pow(p).sum() * ((N * N) if scale_conv else 1)
        elif isinstance(block, Linear):
            lp_loss += block.weight.abs().pow(p).sum()

    return lp_loss


class MyVisionDataset(torchvision.datasets.vision.VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, data=None, targets=None, classes=None,
                 class_to_idx=None, orig_idx=None, sample_weights=None, yield_weights=True, loader=None):
        super(MyVisionDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set
        self.loader = loader

        self.data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data

        if isinstance(data, torch.Tensor) and self.data.ndim == 4 and self.data.shape[2] == self.data.shape[3]:
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = targets.cpu().detach().tolist() if isinstance(targets, torch.Tensor) else targets
        self.targets = targets.tolist() if isinstance(targets, np.ndarray) else targets

        sample_weights = len(targets)*[1.] if sample_weights is None else sample_weights
        self.sample_weights = sample_weights.cpu().detach().tolist() if isinstance(sample_weights, torch.Tensor) else sample_weights
        self.sample_weights = sample_weights.tolist() if isinstance(sample_weights, np.ndarray) else sample_weights
        self.yield_weights = yield_weights

        if orig_idx is None:
            self.orig_idx = np.arange(0, len(targets))
        else:
            self.orig_idx = orig_idx

        self.classes = classes
        self.class_to_idx = class_to_idx

    @staticmethod
    def from_idx(dataset, idx, sample_weights=None, train=True):
        new_len = len(idx) # For debugmode
        new_data = np.array([v[0] for v,i in zip(dataset.samples[:new_len],idx) if i]) if not hasattr(dataset,"data") \
                   else dataset.data[:new_len][idx]
        new_targets = np.array(dataset.targets)[:new_len][idx].tolist()
        new_weights = None if (not hasattr(dataset,"sample_weights") or dataset.sample_weights is None) else \
                      np.array(dataset.sample_weights)[:new_len][idx].tolist()
        yield_weights = False if not hasattr(dataset,"yield_weights") else dataset.yield_weights
        old_orig_idx = dataset.orig_idx if hasattr(dataset,"orig_idx") else np.arange(0,len(dataset))
        new_orig_idx = old_orig_idx[:new_len][idx]
        loader = dataset.loader if hasattr(dataset, "loader") else None
        new_dataset = MyVisionDataset(dataset.root, train, dataset.transform, dataset.target_transform, new_data,
                                      new_targets, dataset.classes, dataset.class_to_idx, new_orig_idx, new_weights,
                                      yield_weights, loader)
        if sample_weights is not None:
            new_dataset.set_weights(sample_weights)
        return new_dataset

    @staticmethod
    def from_idx_and_targets(dataset, idx, new_targets, classes, sample_weights=None):
        new_len = len(idx) # For debugmode
        new_data = np.array([v[0] for v,i in zip(dataset.samples[:new_len],idx) if i]) if not hasattr(dataset,"data") \
                   else dataset.data[:new_len][idx]
        assert new_data.shape[0] == len(new_targets)
        new_targets = new_targets.cpu().detach().numpy() if isinstance(new_targets,torch.Tensor) else new_targets
        new_weights = None if (not hasattr(dataset,"sample_weights") or dataset.sample_weights is None) else \
                      np.array(dataset.sample_weights)[:new_len][idx].tolist()
        yield_weights = False if not hasattr(dataset,"yield_weights") else dataset.yield_weights

        class_to_idx = {int(k) : classes[i] for i,k in enumerate(np.unique(np.array(new_targets)))}
        old_orig_idx = dataset.orig_idx if hasattr(dataset,"orig_idx") else np.arange(0,len(dataset))
        new_orig_idx = old_orig_idx[:new_len][idx]
        loader = dataset.loader if hasattr(dataset,"loader") else None
        new_dataset = MyVisionDataset(dataset.root, dataset.train, dataset.transform, dataset.target_transform, new_data,
                               new_targets, classes, class_to_idx, new_orig_idx, new_weights, yield_weights, loader)
        if sample_weights is not None:
            new_dataset.set_weights(sample_weights)
        return new_dataset

    def set_weights(self, sample_weights=None, yield_weights=True):
        sample_weights = len(self.targets) * [1.] if sample_weights is None else sample_weights
        self.sample_weights = sample_weights.cpu().detach().tolist() if isinstance(sample_weights,
                                                                                   torch.Tensor) else sample_weights
        self.sample_weights = sample_weights.tolist() if isinstance(sample_weights, np.ndarray) else sample_weights
        self.yield_weights = yield_weights


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.loader is not None:
            img = self.loader(img)
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.yield_weights:
            weight = self.sample_weights[index]
            return img, target, weight

        return img, target

    def __len__(self):
        return len(self.data)


def get_scaled_eps(args, layers, relu_ids, curr_layer_idx, j):
    if args.eps_scaling_mode == "COLT":
        eps = args.eps_factor ** (len(layers) - 2 - j) * (args.start_eps_factor * args.train_eps)
    elif args.eps_scaling_mode == "depth":
        depth = len([x for x in relu_ids if x > curr_layer_idx])
        eps = args.eps_factor ** depth * (args.start_eps_factor * args.train_eps)
    elif args.eps_scaling_mode in ["none", "None", None]:
        eps = args.train_eps
    else:
        assert False, "eps scaling mode %s is not known" % args.eps_scaling_mode
    return eps


def reset_params(args, net, dtype, reset=True):
    """ Resets DeepZ slope parameters to the original values. """
    relu_params = []
    for param_name, param_value in net.named_parameters():
        if 'deepz' in param_name:
            relu_params.append(param_value)
            if args.test_domain == 'zono_iter' and reset:
                param_value.data = -torch.ones(param_value.size()).to(param_value.device, dtype=dtype)
            param_value.requires_grad_(True)
        else:
            param_value.requires_grad_(False)
    return relu_params

def get_layers(train_mode, cnet, n_attack_layers=1, min_layer=-2, base_layers=True, protected_layers=0):
    ### return n layers to be attacked plus one previous "warm up" layer prepended
    ### if base_layers use [-2,-1] to prepend, else use layers in the order of their occurence
    relaxed_net = cnet.relaxed_net if "relaxed_net" in dir(cnet) else cnet.module.relaxed_net
    relu_ids = relaxed_net.get_relu_ids()
    if "COLT" in train_mode:
        attack_layers = [x for x in ([-1] + relu_ids) if x >= min_layer]
        if protected_layers > 0:
            attack_layers = attack_layers[:-min(protected_layers,len(attack_layers))]
        warmup_layers = ([-2,-1] if min_layer>=0 else [-2]) \
                        if base_layers \
                        else ([-2] + [x for x in ([-1] + relu_ids) if x < min_layer])[-max(1,1+n_attack_layers-len(attack_layers)):]
        layers = warmup_layers + attack_layers
        layers = layers[:min(n_attack_layers+1, len(layers))]
        assert len(layers) >= 2, "Not enough layers remaining for COLT training"
    elif "adv" in train_mode:
        layers = [-1, -1]
    elif "nat" in train_mode:
        layers = [-2, -2]
    elif "diffAI" in train_mode:
        layers = [-2, -2]
    else:
        raise RuntimeError(f"Unknown train mode {train_mode:}")
    return layers