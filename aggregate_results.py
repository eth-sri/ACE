import functools
import numpy as np
import csv
import re
import os
import argparse
import pickle
from collections.abc import Iterable

def read_csv_to_dict(csv_file):
    with open(csv_file,"r") as f:
        reader = csv.reader(f)
        data_raw = list(reader)
    labels=data_raw[0][0].rstrip(";").split(";")
    if "total" in data_raw[-1][0]:
        data_raw.pop(-1)
    data_np = np.stack([np.array(data_raw[i+1][0].rstrip(";").split(";"),dtype=float) for i in range(len(data_raw)-1)],axis=1)
    data = {k:v for k,v in zip(labels,data_np)}
    return data


def get_accuracy(data_list, mode="nat"):
    assert len(data_list) == 2
    n_exits = len(data_list)
    min_length = functools.reduce(lambda a, b: min(a, b), [len(x["img_id"]) for x in data_list])
    reachability = np.ones((n_exits, min_length), dtype=bool)
    if mode == "nat":
        ok_keys = n_exits * ["nat_ok"]
        for i in range(n_exits):
            reachability[i - 1] = data_list[0]["nat_branch"] == i - 1
    elif mode.startswith("adv"):
        ok_keys = ["pgd_ok_-1" if "pgd_ok_-1" in x.keys() else "pgd_ok" for x in data_list]
        if mode.endswith("adv"):
            reach_pattern = "branch_([0-9]+)_adv_p"
            trunk_key = "branch_-1_adv_p"
        elif mode.endswith("cert"):
            reach_pattern = "branch_([0-9]+)_p"
            trunk_key = "branch_-1_p"
        last_exit = 0
        for data in data_list[:-1]:
            reachable_keys = [x for x in data.keys() if re.match(reach_pattern, x) is not None]
            for i, r_key in enumerate(reachable_keys):
                branch = last_exit + int(re.match(reach_pattern, r_key).group(1))
                reachability[branch] = (data[r_key] != 0).__and__(reachability[branch])
            reachability[branch + 1:] = reachability[branch + 1:].__and__(data[trunk_key] != 0)

    ok = np.stack([data[x] for x, data in zip(ok_keys, data_list)], axis=0).astype(bool)
    ok = (~(((~ok) * reachability).any(axis=0))).astype(int)
    return ok, reachability


def prepare_data(path_list):
    data_list = [read_csv_to_dict(x) for x in path_list]
    min_length = functools.reduce(lambda a, b: min(a, b), [len(x["img_id"]) for x in data_list])
    data_list = [{k: v[0:min_length] for k, v in x.items()} for x in data_list]
    return data_list

def aggregate_accuracies(path_list, model_dir, milp_cert_list=None):
    data_list = prepare_data(path_list)
    n_exits = len(data_list)
    exits = list(range(n_exits-1)) + [-1]

    ver_keys = [x for x in data_list[0].keys() if re.match("ver_ok[_,a-z]",x) is not None]

    nat_ok, nat_reachability = get_accuracy(data_list, mode="nat")
    nat_branch = np.nonzero(nat_reachability.transpose())[1]
    nat_branch[nat_branch == n_exits - 1] = -1
    adv_ok_adv, adv_reachability_adv = get_accuracy(data_list, mode="adv_adv")
    adv_reachability_adv = adv_reachability_adv.astype(int)
    adv_ok_cert, adv_reachability_cert = get_accuracy(data_list, mode="adv_cert")
    adv_reachability_cert = adv_reachability_cert.astype(int)
    ver = np.stack([data_list[0][x] for x in ver_keys],axis=0).any(axis=0).astype(int)
    if milp_cert_list is not None:
        if adv_reachability_cert.shape[0]:
            ver = np.array(milp_cert_list)*(adv_reachability_cert[1]==0)
        else:
            assert False, "MILP cert list only available for individual certification nets"

    csv_log = open(os.path.join(model_dir, "cert_aggregate_log.csv"), mode='w')
    log_writer = csv.writer(csv_log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    log_writer.writerow(["img_id", "label", "nat_ok", "pgd_ok_adv", "pgd_ok_cert", "ver", "nat_branch"]
                        + ["branch_%d_p" % i for i in exits]
                        + ["branch_%d_adv_p" % i for i in exits])

    for i in range(len(nat_ok)):
        log_writer.writerow([data_list[0]["img_id"][i], data_list[0]["label"][i], nat_ok[i], adv_ok_adv[i], adv_ok_cert[i], ver[i], int(nat_branch[i])]
                            + list(adv_reachability_adv[:,i])
                            + list(adv_reachability_cert[:,i]))

    print(f"nat_ok: {nat_ok.mean():.4f}, adv_ok(adv/cert): {adv_ok_adv.mean():.4f}/{adv_ok_cert.mean():.4f}, cert_ok = {ver.mean():.4f}")

    csv_log.close()


def get_args():
    parser = argparse.ArgumentParser(description='Aggregate results for ACE quick evaluation.')

    # Basic arguments
    parser.add_argument('--core-file', default='None', type=str, required=True, help='Path to the test log of the core network')
    parser.add_argument('--certification-file', default='None', required=True, help='Path to the cert_log of the ACE network without core')
    parser.add_argument('--milp-cert', default=False, action="store_true", required=False)
    args = parser.parse_args()

    return args

def get_obj_value(x,field):
    x = pickle.load(open(x,'rb'))
    y = []
    for k in x.keys():
        if field == k:
            y+=[x[field]]
    for v in x.values():
        if isinstance(v,Iterable):
            if field in v:
                y += [v[field]]
    return y

def main():
    args = get_args()
    model_dir = args.certification_file.rstrip("/cert_log.csv")
    path_list = [args.certification_file, args.core_file]


    if args.milp_cert:
        ver_folder_path = "models_new/cifar10/convmedbig_flat_2_2_4_250_0.00784/1579280297/6/net_800_ver"
        milp_cert = [(lambda x: int(x[0]) if len(x) > 0 else 0)(get_obj_value("%s/%d.p" % (ver_folder_path, i), 'verified'))
             for i in range(int(1e4))]
    else:
        milp_cert = None

    # aggregate_accuracies(path_list, model_dir, None)

    aggregate_accuracies(path_list, model_dir, milp_cert)


if __name__ == '__main__':
    main()