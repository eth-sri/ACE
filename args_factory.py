import argparse
import re

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def translate_net_name(net_name):
    match_C = re.match("C([0-9]+)_([a-z,0-9]+)", net_name)
    match_W = re.match("WRN_([a-z,0-9]+)", net_name)

    new_net_name = net_name

    if match_C is not None:
        dataset = match_C.group(2)
        conv_id = int(match_C.group(1))

        if "image" in dataset:
            if conv_id == 3:
                new_net_name = "myNet__2_4_8__5__2__2__0__0__1"
            elif conv_id == 5:
                new_net_name = "myNet__4_4_8_8_8__3__1_1_2_1_1__1__0__512"
        elif "cifar" in dataset:
            if conv_id == 2:
                new_net_name = "myNet__2_2__4__2__0__200"
            elif conv_id == 3:
                new_net_name = "convmedbig_flat_2_2_4_250"
            elif conv_id == 5:
                new_net_name = "myNet__4_4_8_8_8__3__1_1_2_1_1__1__0__512"
    elif match_W is not None:
        dataset = match_W.group(1)
        if "image" in dataset:
            new_net_name = "myresnet_wide_1_1_1_w10_p7"
    return new_net_name


def get_args():
    parser = argparse.ArgumentParser(description='ACE compositional architectures for boosting certified robustness.')
    
    # Basic arguments
    parser.add_argument('--train-mode', default='train', type=str, help='whether to train adversarially')
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--train-batch', default=100, type=int, help='batch size for training')
    parser.add_argument('--test-batch', default=100, type=int, help='batch size for testing')
    parser.add_argument('--n-epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--test-freq', default=50, type=int, help='frequency of testing')
    parser.add_argument('--test-set', default="test", help='Run zono-certify after training')
    parser.add_argument('--debug', action='store_true', help='Stop train and test iterations early')
    parser.add_argument('--cert', action='store_true', help='Run zono-certify after training')
    parser.add_argument('--cert-trunk', default=True, type=boolean_string, help="Whether to attempt to certify the trunk")
    parser.add_argument('--num-workers', default=16, type=int, help="Number of workers for data loaders")

    #model-loading
    parser.add_argument('--load-model', default=None, type=str, help='model to load')

    # Optimizer and learning rate scheduling
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer to use')
    parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
    parser.add_argument('--lr-sched', default='step_lr', type=str, choices=['step_lr', 'cycle'], help='choice of learning rate scheduling')
    parser.add_argument('--lr-step', default=10, type=int, help='number of epochs between lr updates')
    parser.add_argument('--lr-factor', default=0.5, type=float, help='factor by which to decrease learning rate')
    parser.add_argument('--lr-layer-dec', default=0.5, type=float, help='factor by which to decrease learning rate in the next layers')
    parser.add_argument('--pct-up', default=0.5, type=float, help='percentage of cycle to go up')

    # Losses and regularizers
    parser.add_argument('--nat-factor', default=0.0, type=float, help='factor for natural loss')
    parser.add_argument('--relu-stable-cnt', default=1, type=int, help='number of relus to stabilize in front')
    parser.add_argument('--relu-stable-type', required=False, type=str, choices=['tight', 'tanh', 'mixed'], default='tight', help='stability regularizer type')
    parser.add_argument('--relu-stable', required=False, type=float, default=None, help='factor for relu stability')
    parser.add_argument('--relu-stable-layer-dec', required=False, type=float, default=1.0, help='factor for relu stability')
    parser.add_argument('--l1-reg', default=0.0, type=float, help='l1 regularization coefficient')
    parser.add_argument('--l2-reg', default=0.0, type=float, help='l2 regularization coefficient')
    parser.add_argument('--reg-scale-conv', default=False, type=boolean_string, help="scale regularization term of conv weights by the number of their application")
    parser.add_argument('--mix', action='store_true', help='whether to mix adversarial and standard loss')
    parser.add_argument('--mix-epochs', default=0, type=int, help='number of epochs to anneal schedule')
    parser.add_argument('--relu-stable-protected', default=1, type=int, help="How many layers to attack without ReLU stable loss")
    parser.add_argument('--n-attack-layers', default=4, type=int, help="Number of latent spaces to be attacked. 1 => adv training")
    parser.add_argument('--protected-layers', default=1, type=int, help="how many layers prior to classification not to attack")


    # Configuration of adversarial attacks
    parser.add_argument('--train-eps', default=0.0, type=float, help='epsilon to train with')
    parser.add_argument('--adv-type', default="pgd", type=str, help='type of adv attack')
    parser.add_argument('--test-eps', default=None, type=float, help='epsilon to verify')
    parser.add_argument('--anneal', action='store_true', help='whether to anneal epsilon')
    parser.add_argument('--anneal-epochs', default=1, type=int, help='number of epochs to anneal eps over')
    parser.add_argument("--anneal-pow", default=1, type=float, help="power law schedule for annealing")
    parser.add_argument("--anneal-warmup", default=0, type=int, help="Use warmup for epsilon annealing")
    parser.add_argument('--eps-factor', default=1.05, type=float, help='factor to increase epsilon per layer')
    parser.add_argument('--eps-scaling-mode', default="COLT", type=str, choices=["COLT","depth"], help="Use number of following COLT layers or depth to compute eps scaling")
    parser.add_argument('--start-eps-factor', default=1.0, type=float, help='factor to determine starting epsilon')
    parser.add_argument('--train-att-n-steps', default=8, type=int, help='number of steps for the attack')
    parser.add_argument('--train-att-step-size', default=0.25, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--test-att-n-steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--test-att-step-size', default=None, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--n-rand-proj', default=50, type=int, help='number of random projections')
    parser.add_argument('--beta-start', default=0., type=float, help='initial kappa')
    parser.add_argument('--beta-end', default=1., type=float, help='final kappa')



    # Metadata
    parser.add_argument('--exp-name', default='dev', type=str, help='name of the experiment')
    parser.add_argument('--exp-id', default=1, type=int, help='name of the experiment')
    parser.add_argument('--no-cuda', action='store_true', help='whether to use only cpu')
    parser.add_argument('--root-dir', required=False, default='./', type=str, help='directory to store the data')

    ## SpecNet - Architecture
    parser.add_argument('--n-branches', default=1, type=int, help="Number of branches")
    parser.add_argument('--train-trunk', default=True, type=boolean_string, help="Allow retraining of the trunk")
    parser.add_argument('--branch-nets', default="myNet__2_2_4__3__2__0__0__100", type=str, nargs="+", help="Specify branch net architectures")
    parser.add_argument('--gate-nets', default=None, type=str, nargs="+", help="Specify gate net architectures")
    parser.add_argument('--cert-net-dim', default=None, type=int, help="Size to scale input image to for branch and gate nets")

    ## SpecNet-Loading
    parser.add_argument('--load-branch-model', default=None, type=str, nargs="+", help="Model to load on branches. 'True' will load same model as on trunk ")
    parser.add_argument('--load-gate-model', default=None, type=str, nargs="+", help="Model to load on branches. 'True' will load same model as on trunk ")
    parser.add_argument('--load-trunk-model', default=None, type=str, help="model to load on trunk")
    parser.add_argument("--gate-feature-extraction", default=None, type=int, help="How many (linear) layers to retrain at the end of a loaded gate net")

    ## SpecNet - Gating Objective
    parser.add_argument('--gate-type', default="net", choices=["net", "entropy"], help="Chose whether gating should be based on entropy or a network")
    parser.add_argument('--gate-mode', default="gate_initial", help="Chose gate target initializiation: gate_initial") # gate_initial, gate_c_mat,
    parser.add_argument('--gate-threshold', default=0.0, type=float, help="Threshold for gate network selection. Entropy is negative.")
    parser.add_argument('--gate-on-trunk', default=True, type=boolean_string, help="Train gate using trunk or branch to determine targets")
    parser.add_argument('--retrain-branch', default=False, type=boolean_string, help="Retrain branch with weighted dataset after gate training, only if gate_on_trunk is False")

    ## SpecNet - Gate Training
    parser.add_argument('--exact-target-cert', action="store_true", help="Use exact certificaiton for selection of targets for gate networks")
    parser.add_argument('--cotrain-entropy', default=None, type=float, help="weighting factor of entropy gate loss compared to branch")
    parser.add_argument('--balanced-gate-loss', action="store_true", help="use balanced loss to train gate networks")
    parser.add_argument('--sliding-loss-balance', default=None, type=float, help="begin with balanced loss, moving towards unbalanced loss")
    parser.add_argument('--gate-target-type', default="nat", type=str, choices=["nat","equal"], help="Gate target for Entropy selection.")

    ##DiffAI
    parser.add_argument('--train-domain', default=None, type=str, nargs="+", help="Specify the domains to be used during DiffAI training")
    parser.add_argument('--train-domain-weights', default=None, type=float, nargs="+", help="If multiply train domains are specified, set weight")
    parser.add_argument('--cert-domain', default="zono", type=str, nargs="+",help="Specify domains to certify wiht")

    ## Weighted Training
    parser.add_argument("--weighted-training", type=float, default=None, help="Factor to reduce sample weight by instead of removing it from training set")

    args = parser.parse_args()

    if args.train_mode == "train":
        args.train_mode = "train-COLT"

    if args.test_eps is None:
        args.test_eps = args.train_eps
    if args.test_att_n_steps is None:
        args.test_att_n_steps = args.train_att_n_steps
        args.test_att_step_size = args.train_att_step_size

    if args.weighted_training == 0:
        args.weighted_training = None

    if args.gate_type == "net":
        args.cotrain_entropy = None
    elif args.gate_type == "entropy":
        args.gate_on_trunk = True

    if args.gate_on_trunk:
        args.retrain_branch = True

    if args.load_branch_model == "True":
        args.load_branch_model = args.load_trunk_model

    if args.gate_type != "net":
        args.gate_feature_extraction = None
        print("No selection network was loaded, but feature extraction activated!")

    if args.debug:
        args.num_workers=0

    if not isinstance(args.train_domain,list): args.train_domain = [args.train_domain]
    if not isinstance(args.cert_domain, list): args.cert_domain = [args.cert_domain]
    if args.train_domain_weights is None:
        args.train_domain_weights = [1. for _ in args.train_domain]
    weight_sum = sum(args.train_domain_weights)
    args.train_domain_weights = {domain: x/weight_sum for domain,x in zip(args.train_domain, args.train_domain_weights)}

    args.branch_nets = [translate_net_name(net) for net in args.branch_nets] if isinstance(args.branch_nets,list) else translate_net_name(args.branch_nets)
    args.gate_nets = None if args.gate_nets is None else [translate_net_name(net) for net in args.gate_nets]
    args.net = translate_net_name(args.net)

    return args
