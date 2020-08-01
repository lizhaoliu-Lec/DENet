from .schedulers import ConstantLR
from .schedulers import PolynomialLR
from .schedulers import MultiStepLR
from .schedulers import CosineAnnealingLR
from .schedulers import ExponentialLR

key2scheduler = {
    'ConstantLR': ConstantLR,
    'PolynomialLR': PolynomialLR,
    'MultiStepLR': MultiStepLR,
    'CosineAnnealingLR': CosineAnnealingLR,
    'ExponentialLR': ExponentialLR
}


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        print('Using no lr scheduler')
        return ConstantLR(optimizer)

    s_type = scheduler_dict['name']
    scheduler_dict.pop('name')

    print('Using scheduler: {} with {} params'.format(s_type, scheduler_dict))

    return key2scheduler[s_type](optimizer, **scheduler_dict)
