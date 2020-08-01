import functools

from .losses import cross_entropy2d
from .losses import loss_general
from .losses import loss_denet

key2loss = {
    'loss_general': loss_general,
    'loss_denet': loss_denet,
}


def get_loss(loss_dict):
    if loss_dict is None:
        print("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_name = loss_dict['name']
        loss_params = {k: v for k, v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        print('Using loss: {} with params {}'.format(loss_name,
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
