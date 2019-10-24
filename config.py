from argparse import ArgumentParser
import datetime
from pytz import timezone
import sys
import os

tz = timezone("Europe/Berlin")

# ifr experiment


class DefaultArguments:
    name = 'trash'
    loss_fct = 'soft_ce'
    real_exp = False

    dataset_modification = 'None'   # single_feat, double_feat, shift_mnist

    # training
    model = 'cnn'
    training_mode = 'normal'       # append_dataset, normal, overwrite, adversarial_training (=DDN), 'siamese_adversarial_training'
    reinit_network = True
    percentage_to_append = 0.2           #
    n_loops = 10                    # DDN: 50
    n_epochs = 10                   # DDN: 1
    batch_size = 128                # DDN: 128
    weight_decay = 1e-6            # DDN: 1e-6

    batch_size_test = 1000

    # optimizer
    optimizer = 'sgd'      # DDN: sgd
    lr = 0.01               # DDN: 0.01
    momentum = 0.9          # DDN: 0.9
    lr_step = 30             # DNN: 30
    lr_decay = 0.1

    # attack
    attacks_names = ['BIM', 'PGD_1.5', 'PGD_2.0', 'PGD_2.5', 'PGD_3.0', 'DDN_L2']
    attack_for_new_dataset = 'DDN_L2'
    max_eps_new_dataset = 10.
    lp_metric = 'l2'
    attack_batch_size = 1000
    attack_iter = 80
    attack_l2_step_size = 0.05
    attack_linf_step_size = 0.05
    epsilon_accuracy_l2 = 2.0
    epsilon_accuracy_linf = 0.3
    epsilon_max_l2 = 10.

    # adv training
    adv_train_attack_name = 'PGD'    # DDN: DDN_L2
    adv_train_epsilon = 1.5          # 2.4
    adv_attack_iter = 20              # DDN: 100
    adv_l2_step_size = 0.05
    # adv_epsilon = 2.

    n_classes = 10
    end = 60000


def parse_arguments(**passed_args):
    print('passed args', passed_args)

    default_arguments = class_to_dict(DefaultArguments)
    if passed_args:
        default_arguments = {**default_arguments, **passed_args}

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

    args = args.__dict__
    for key, val in args.items():
        if val == 'false' or val == 'False':
            args[key] = False
        if val == 'true' or val == 'True':
            args[key] = True

    # change some args
    if args['dataset_modification'] == 'double_feat' or args['dataset_modification'] == 'single_feat':
        print('changed end')
        args['end'] = 1000
        args['n_epoch'] = 20

    if args['lr_step'] == 0:
        args['lr_step'] = args['n_epoch']

    args['exp_name'] = get_run_name(default_arguments, args)

    proj_dir = os.getcwd()
    if args['real_exp']:
        exp_folder = proj_dir + '/exp_fr/'
    else:
        exp_folder = proj_dir + '/test_exp/'
    args['experiment_folder'] = exp_folder + args['exp_name']

    print('Parsed arguments:')
    args = AttrDict(args)
    args.proj_dir = proj_dir
    print('args', args)
    return args


def get_run_name(default_args, args):
    exp_name = args['name'] + '_'
    for key in default_args:
        old_val = default_args[key]
        if old_val != args[key] and key != 'device' \
                and key != 'name' and key != 'expfolder':
            val = args[key]
            if isinstance(val, float):
                exp_name += f'{key}{val:.3f}-'
            elif isinstance(val, str):
                exp_name += f'{key}' + val[:5] + '-'
            else:
                exp_name += f'{key}' + str(val) + '-'

    tz = timezone("Europe/Berlin")
    return exp_name + '--' + datetime.datetime.now(tz=tz).strftime(
        "%Y-%m-%d-%H-%M-%S")


def class_to_dict(o):
    keys = [f for f in dir(o) if not callable(getattr(o, f)) and not f.startswith('__')]
    new_dict = {}
    for key in keys:
        new_dict[key] = o.__dict__[key]
    return new_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



