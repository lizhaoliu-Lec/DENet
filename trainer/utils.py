import sys
import time
import functools
import torch
import numpy as np
import torch.nn.functional as F

from model.head import key2arch
from utils.operations import decode_seg_map, get_binary_logits
from utils import sort_dict_by

arch2files = {
    'CAN': "model/head/can.py",
    'PGN': "model/head/pgn.py",
    'AMP': "model/head/amp.py",
    'DENet': "model/head/denet.py",
}


class Timer(object):
    """The class for timer."""

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Name2Metric(object):
    def __init__(self, name, metric):
        self.metric = metric
        self.name = name
        self.name2metric = {n: m() for n, m in zip(name, metric)}

    def reset(self):
        name, metric = self.name, self.metric
        self.name2metric = {n: m() for n, m in zip(name, metric)}

    def __getitem__(self, item):
        return self.name2metric[item]

    def __setitem__(self, key, value):
        self.name2metric[key] = value


class Name2Meter(object):
    def __init__(self, name, meter):
        self.meter = meter
        self.name = name
        self.name2meter = {n: l() for n, l in zip(name, meter)}

    def reset(self):
        name, meter = self.name, self.meter
        self.name2meter = {n: m() for n, m in zip(name, meter)}

    def __getitem__(self, item):
        return self.name2meter[item]

    def __setitem__(self, key, value):
        self.name2meter[key] = value


def center_print(content, around='*', repeat_around=10):
    num = repeat_around
    s = around
    print(num * s + ' %s ' % content + num * s)


def get_sorted_dict_str(d):
    l = sort_dict_by(d)
    d_str = '{' + ', '.join([str(k) + ':' + str(round(v, 4))
                             if v != 0 else str(k) + ':' + '0.0' for (k, v) in l]) + '}'
    return d_str


"""Utilities for plotting figures."""


def mask_gray(mask, pre=True):
    return mask.data.max(0)[1].cpu().numpy() if pre else mask.squeeze().cpu().numpy()


def mask_color(mask, num_classes, pre=True):
    color_mask = mask.data.max(0)[1].cpu().numpy() if pre else mask.squeeze().cpu().numpy()
    return decode_seg_map(color_mask, num_classes=num_classes)


def query_rgb(query):
    query = query.squeeze(0).permute([1, 2, 0]).cpu().numpy()
    return (query - np.min(query)) / (np.max(query) - np.min(query))


def upsample(Yq_pre, Yq):
    if Yq_pre.size()[2:] != Yq.size()[1:]:
        Yq_pre = F.interpolate(Yq_pre, size=Yq.size()[1:], mode='bilinear', align_corners=True)
    return Yq_pre


def process_nway(Yq_pre, way):
    """
    Helper function for processing n-way predictions of class-agnostic models.

    Args:
        Yq_pre: a tensor with shape (N, 2 * way, h, w).
        way: an integer specifies the way.

    Returns:
        the processed n-way segmentation mask with shape (N, h, w).
    """

    N = Yq_pre.shape[0]
    background = Yq_pre.index_select(
        dim=1, index=torch.tensor([2 * _ for _ in range(way)], device=Yq_pre.device))
    background = torch.max(background, dim=1, keepdim=True)[0]
    foreground = Yq_pre.index_select(
        dim=1, index=torch.tensor([2 * _ + 1 for _ in range(way)], device=Yq_pre.device))
    # N, way + 1, h, w
    scores = torch.cat([background, foreground], dim=1)
    binary_scores = []
    for i in range(way):
        l = torch.full([N], i + 1, dtype=torch.int64, device=Yq_pre.device)
        binary_scores.append(get_binary_logits(scores, l))
    # N, way, 2, h, w
    binary_scores = torch.stack(binary_scores, dim=1)
    return binary_scores


def to_cuda(*args):
    def general_cuda(obj):
        if isinstance(obj, list):
            return [o.cuda() for o in obj]
        else:
            return obj.cuda()

    return [general_cuda(a) for a in args]


def knowledge_loss_wrapper(loss_func, arch):
    if arch is None:
        raise ValueError('Arch name must specify!')
    if arch not in key2arch:
        raise NotImplementedError('Architecture {} not implemented'.format(arch))

    @functools.wraps(loss_func)
    def wrapper(*args, **kwargs):
        if arch in ['AMP']:
            Y_pre, Y = args[0:2]
            return loss_func(Y_pre, Y, **kwargs)
        else:
            raise ValueError('Architecture {} not supported'.format(arch))

    return wrapper


def knowledge_forward_wrapper(model, arch, *args, **kwargs):
    if arch is None:
        raise ValueError('Arch name must specify!')

    if arch not in key2arch:
        raise NotImplementedError('Architecture {} not implemented'.format(arch))

    if arch in ['AMP']:
        if model.training:
            I = args[2]
            return model(I, **kwargs), None
        else:
            Is, Ys, Iq, _, cls = args
            return model(Is, Ys, Iq, cls, **kwargs), None
    else:
        raise ValueError('Architecture {} not supported'.format(arch))


def pattern_loss_wrapper(loss_func, arch):
    """
    Loss interfaces:
    Given input: Yq_pre, Yq, Ys_pre, Ys.
    Output: Loss.
    """

    if arch is None:
        raise ValueError('Arch name must specify!')

    if arch not in key2arch:
        raise NotImplementedError('Architecture {} not implemented'.format(arch))

    @functools.wraps(loss_func)
    def wrapper(*args, **kwargs):

        if arch in ['SGN', 'CAN', 'PGN', 'FWB']:
            Yq_pre, Yq = args[0:2]
            return loss_func(Yq_pre, Yq, **kwargs)
        if arch in ['PAN']:
            Yq_pre, Yq, Ys_pre, Ys = args
            return loss_func(Yq_pre, Yq, **kwargs) + loss_func(Ys_pre, Ys, **kwargs)
        raise ValueError('Architecture {} not supported'.format(arch))

    return wrapper


def pattern_forward_wrapper(model, arch, *args, **kwargs):
    """
    Forward Interfaces:
    Given input: Is, Ys, Iq, Yq.
    Output: Yq_pre, Ys_pre.
    """

    if arch is None:
        raise ValueError('Arch name must specify!')

    if arch not in key2arch:
        raise NotImplementedError('Architecture {} not implemented'.format(arch))

    if arch in ['SGN', 'CAN', 'PGN', 'FWB']:
        Is, Ys, Iq = args[0:3]
        return model(Is, Ys, Iq, **kwargs), None
    if arch in ['PAN']:
        Is, Ys, Iq = args[0:3]
        return model(Is, Ys, Iq, **kwargs)
    raise ValueError('Architecture {} not supported'.format(arch))


def DENet_loss_wrapper(loss_func, arch):
    """
    Loss interfaces:
    Given input: Yq_pre, Yq, Ys_pre, Ys, Yq_full_pre, Yq_full, Ys_full_pre, Ys_full.
    Output: knowledge loss, pattern loss.
    """

    if arch is None:
        raise ValueError('Arch name must specify!')

    if arch not in key2arch:
        raise NotImplementedError('Architecture {} not implemented'.format(arch))

    @functools.wraps(loss_func)
    def wrapper(*args, **kwargs):

        if arch in ['DENet']:
            Yq_pre, Yq, _, _, Yq_full_pre, Yq_full, _, _, weight = args
            # loss k: loss computed with full masks and full logits
            loss_k = 0.0
            batch_size = Yq_full_pre.shape[0]
            for pre, tar, w in zip(
                    Yq_full_pre.chunk(batch_size, dim=0),
                    Yq_full.chunk(batch_size, dim=0),
                    weight):
                loss_k += loss_func(pre, tar, weight=w, mode='knowledge')
            loss_k /= batch_size
            # loss p: loss computed with binary masks and binary logits
            loss_p = loss_func(Yq_pre, Yq, mode='pattern')
            return loss_k, loss_p
        raise ValueError('Architecture {} not supported'.format(arch))

    return wrapper


def DENet_forward_wrapper(model, arch, *args, **kwargs):
    """
    Forward Interfaces:
    Given input: Is, Ys, Iq, Yq, cls_id, Ys_full, Yq_full.
    Output: Yq_pre, Ys_pre, Yq_full_pre, Ys_full_pre.
    """

    if arch is None:
        raise ValueError('Arch name must specify!')

    if arch not in key2arch:
        raise NotImplementedError('Architecture {} not implemented'.format(arch))

    if arch in ['DENet']:
        Is, Ys, Iq, Yq, cls_id, Ys_full, Yq_full = args
        outs = model(Is, Ys, Iq, cls_id, **kwargs)
        return outs[1], None, outs[0], None
    raise ValueError('Architecture {} not supported'.format(arch))


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
