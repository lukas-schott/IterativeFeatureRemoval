from argparse import ArgumentParser
import datetime
from pytz import timezone
import sys

tz = timezone("Europe/Berlin")


class DefaultArguments:
    name = 'trash'
    real_exp = False

    lr = 0.0001         # madry 0.0001
    weight_decay = 0.0  # madry 0
    n_epochs = 100      # madry ~ 83
    batch_size = 128    # madry 50
    loss_fct = 'ce'     # 'ce' or 'soft_ce' madry: ce

    er = 0.

    adv_training = True
    lp_metric = 'linf'
    attack_iter = 40        # madry 40
    adv_epsilon = 0.3       # madry 0.3
    pgd_step_size = 0.01    # madry 0.01

    # dataset
    n_classes = 10
    dataset_modification = 'None'   # single_feat, double_feat, shift_mnist
    end = 60000


def parse_arguments(**passed_args):
    print('passed args', passed_args)

    default_arguments = class_to_dict(DefaultArguments)
    if passed_args:
        default_arguments = {**default_arguments, **passed_args}

    parser = ArgumentParser(
        description='Robustness Through Adversarial Training')
    for key, value in default_arguments.items():
        if isinstance(value, bool):
            value = str(value).lower()
        parser.add_argument('--' + key, default=value, type=type(value))

    # works with ipython
    if 'ipykernel_launcher.py' in sys.argv[0] or 'default_worker.py' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    args = args.__dict__
    # print('args before last', args)
    # convert str to boolean
    for key, val in args.items():
        if val == 'false':
            args[key] = False
        if val == 'true':
            args[key] = True

    # change some args
    if args['dataset_modification'] == 'double_feat' or args['dataset_modification'] == 'single_feat':
        print('changed end')
        args['end'] = 1000
        args['n_epoch'] = 20

    args['exp_name'] = get_run_name(default_arguments, args)

    if args['real_exp']:
        exp_folder = './exp_adv_training/'
    else:
        exp_folder = './test_exp/'
    args['experiment_folder'] = exp_folder + args['exp_name']

    print('Parsed arguments:')
    args = AttrDict(args)
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
