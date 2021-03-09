import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import robustness
import robustness.datasets
import robustness.tools
import re
import socket
import CImageNetRanges
# import pickle
# from collections.abc import Iterable


IMAGENET_DIR = "./data"


def get_mean_sigma(device, dataset):
    if dataset == 'cifar10':
        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view((3, 1, 1))
        sigma = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view((3, 1, 1))
    elif re.match("[r,c]{0,1}imagenet([0-9]*)", dataset) is not None:
        if re.match("[r,c]{1}imagenet([0-9]*)", dataset) is not None:
            ds = robustness.datasets.RestrictedImageNet(os.path.abspath(IMAGENET_DIR))
        else:
            ds = robustness.datasets.ImageNet(os.path.abspath(IMAGENET_DIR))
        mean = ds.mean.view((3, 1, 1))
        sigma = ds.std.view((3, 1, 1))
    elif re.match("timagenet", dataset) is not None:
        mean = torch.FloatTensor([0.4802, 0.4481, 0.3975]).view((3, 1, 1))
        sigma = torch.FloatTensor([0.2302, 0.2265, 0.2262]).view((3, 1, 1))
    elif dataset == "mnist":
        mean = torch.FloatTensor([0.1307]).view((1, 1, 1))
        sigma = torch.FloatTensor([0.3081]).view((1, 1, 1))
    else:
        raise RuntimeError(f"Dataset {dataset:s} is unknown.")
    return mean.to(device), sigma.to(device)


def get_mnist():
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 28, 1, 10


def get_cifar10(debug=False):
    if debug:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train_set, test_set, 32, 3, 10


def get_imagenet(dataset):
    dim = 224 if dataset.endswith("net") else int(re.match("[c,r]{0,1}imagenet([0-9]*)",dataset).group(1))
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_valid = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(dim),
        transforms.ToTensor(),
    ])

    train_dir = os.path.join(IMAGENET_DIR, 'train')
    valid_dir = os.path.join(IMAGENET_DIR, 'val')

    if dataset.startswith("r"):
        ds = robustness.datasets.RestrictedImageNet(os.path.abspath(IMAGENET_DIR))
    elif dataset.startswith("c"):
        _, class_to_idx = robustness.tools.folder.DatasetFolder._find_classes(None, dir=os.path.abspath(train_dir))
        c_imagenet_word_ids = CImageNetRanges.C_IMAGENET_RANGES
        imagenet_ranges = [(class_to_idx[x], class_to_idx[x]) for x in c_imagenet_word_ids]
        ds = robustness.datasets.CustomImageNet(os.path.abspath(IMAGENET_DIR),imagenet_ranges)
    else:
        ds = robustness.datasets.ImageNet(os.path.abspath(IMAGENET_DIR))

    train_set = robustness.tools.folder.ImageFolder(root=train_dir, label_mapping=ds.label_mapping, transform=transform_train)
    valid_set = robustness.tools.folder.ImageFolder(root=valid_dir, label_mapping=ds.label_mapping, transform=transform_valid)

    return train_set, valid_set, dim, 3, ds.num_classes


def get_tinyimagenet():
    train_set = datasets.ImageFolder(IMAGENET_DIR + '/tinyImageNet/tiny-imagenet-200/' + '/train',
                                      transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(56, padding_mode='edge'),
                                          transforms.ToTensor()
                                      ]))
    test_set = datasets.ImageFolder(IMAGENET_DIR + '/tinyImageNet/tiny-imagenet-200/'+ '/val',
                                     transform=transforms.Compose([
                                         transforms.CenterCrop(56),
                                         transforms.ToTensor()]))

    return train_set, test_set, 56, 3, 200

# def get_cifar_MILP_cert():
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),])
#
#     test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#     ver_folder_path = "models_new/cifar10/convmedbig_flat_2_2_4_250_0.00784/1579280297/6/net_800_ver"
#
#     milp_cert = [(lambda x: int(x[0]) if len(x)>0 else 0)(get_obj_value("%s/%d.p" %(ver_folder_path,i),'verified')) for i in range(int(1e4))]
#     test_set.classes = ['no_cert', 'cert']
#     test_set.class_to_idx = {'no_cert': 0, 'cert': 1}
#     test_set.targets = milp_cert
#     return None, test_set, 32, 3, 10


def get_loaders(args):
    if args.dataset == 'cifar10':
        train_set, test_set, input_size, input_channels, n_class = get_cifar10(args.debug)
    # elif args.dataset == 'cifar10_MILP_cert':
    #     train_set, test_set, input_size, input_channels, n_class = get_cifar_MILP_cert()
    elif args.dataset == 'mnist':
        train_set, test_set, input_size, input_channels, n_class = get_mnist()
    elif re.match("[c,r]{0,1}imagenet([0-9]*)",args.dataset) is not None:
        train_set, test_set, input_size, input_channels, n_class = get_imagenet(args.dataset)
    elif args.dataset == 'timagenet':
        train_set, test_set, input_size, input_channels, n_class = get_tinyimagenet()
    else:
        raise NotImplementedError('Unknown dataset')
    args.input_min, args.input_max = 0, 1

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch, shuffle=~args.debug,
                                               num_workers=args.num_workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return len(train_set), train_loader, test_loader, input_size, input_channels, n_class

# def get_obj_value(x,field):
#     x = pickle.load(open(x,'rb'))
#     y = []
#     for k in x.keys():
#         if field == k:
#             y+=[x[field]]
#     for v in x.values():
#         if isinstance(v,Iterable):
#             if field in v:
#                 y += [v[field]]
#     return y