from argparse import ArgumentParser
import datetime
from pytz import timezone
import sys
import os

tz = timezone("Europe/Berlin")

# ifr experiment


class DefaultArguments:
    exp_series = 'trash'
    name = 'trash'
    loss_fct = 'ce'                 # 'soft_ce', 'ce'
    real_exp = False

    # data
    dataset_modification = 'None'   # single_feat, double_feat, shift_mnist
    dataset = 'MNIST'
    mnist_c = False

    # training
    model = 'cnn'
    training_mode = 'normal'       # normal, adversarial, adversarial_projection, redundancy
    activation_matching = False
    # old append_dataset, project_and_activation_matching, normal, project_out, adversarial_training (=DDN), 'siamese_adversarial_training'
    n_redundant = 1
    model_load_path = ''

    # reinit_network = True

    # redundancy networks
    train_greedily = False
    cosine_dissimilarity_weight = 0.1

    # percentage_to_append = 0.2           #
    accuracy_eval_interval = 1
    n_epochs = 50                   # DDN: 50
    batch_size = 128                # DDN: 128
    weight_decay = 1e-6            # DDN: 1e-6
    siamese_activations_weight = 1
    batch_size_test = 1000

    # optimizer
    optimizer = 'sgd'      # DDN: sgd
    lr = 0.01               # DDN: 0.01
    momentum = 0.9          # DDN: 0.9
    lr_step = 30             # DNN: 30
    lr_decay = 0.1

    # attacks
    lp_metric = 'l2'
    attack_batch_size = 1000

    # attack eval
    robustness_eval_interval = 5
    attacks_eval_names = ['BIM', 'PGD_1.5', 'PGD_2.0', 'PGD_2.5', 'PGD_3.0', 'DDN_L2']
    attack_eval_max_eps_l2 = 10
    attack_eval_iter = 100                     # DDN 300
    attack_eval_l2_step_size = 0.05
    attack_eval_linf_step_size = 0.01
    epsilon_threshold_accuracy_l2 = 1.5
    epsilon_threshold_accuracy_linf = 0.3

    # attack train
    epoch_start_adv_train = 0
    attack_train_name = 'DDN_L2'  # PGDd.d, BIM, DDN_L2
    attack_train_max_eps_l2 = 2.4
    attack_train_iter = 100
    attack_train_l2_step_size = 0.05
    attack_percentage_clean = 0.
    # adv training

    n_classes = 10
    end = 100000


def parse_arguments(passed_args=None):
    # print('passed args', passed_args)
    default_arguments = c2d(DefaultArguments)

    parser = ArgumentParser(description='Robustness Through Adversarial Training')
    for key, value in default_arguments.items():
        if isinstance(value, bool):
            value = str(value).lower()
        parser.add_argument('--' + key, default=value, type=type(value))

    # fix to work with ipython
    if 'ipykernel_launcher.py' in sys.argv[0] or 'default_worker.py' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    if passed_args:
        args = {**args.__dict__, **passed_args}
    else:
       args = args.__dict__  # convert to dct to iterate and parse

    for key, val in args.items():
        if val == 'false' or val == 'False':
            args[key] = False
        if val == 'true' or val == 'True':
            args[key] = True
    args = AttrDict(args)  # convert back to class because its more convenient to write . insted of ['...']

    # change some args
    if args.dataset_modification == 'double_feat' or args.dataset_modification == 'single_feat':
        print('changed end')
        args.end = 1000
        args.n_epoch = 20

    if args.lr_step == 0:
        args.lr_step = args.n_epoch

    args.exp_name = get_run_name(c2d(DefaultArguments), args)
    proj_dir = os.getcwd()
    exp_folder = proj_dir + '/exp/' + args.exp_series + '/'

    args.experiment_folder = exp_folder + args.exp_name

    args = AttrDict(args)
    args.proj_dir = proj_dir
    if args.training_mode != 'redundancy':
        args.n_redundant = 1
    print('args', args)
    print('exp name', args.exp_name)
    return args


def get_run_name(default_args, args):
    exp_name = args['name'] + '_'
    default_args.pop('exp_series')
    for key in default_args:
        if len(exp_name) > 100:
            break

        old_val = default_args[key]
        if old_val != args[key] and key != 'device' \
                and key != 'name' and key != 'expfolder':
            val = args[key]
            if isinstance(val, float):
                exp_name += f'{key}{val:.2f}-'
            elif isinstance(val, str):
                exp_name += f'{key}' + val[:5] + '-'
            else:
                exp_name += f'{key}' + str(val) + '-'


    tz = timezone("Europe/Berlin")
    return exp_name + '--' + datetime.datetime.now(tz=tz).strftime(
        "%Y-%m-%d-%H-%M-%S")


def c2d(o):
    keys = [f for f in dir(o) if not callable(getattr(o, f)) and not f.startswith('__')]
    new_dict = {}
    for key in keys:
        new_dict[key] = o.__dict__[key]
    return new_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



