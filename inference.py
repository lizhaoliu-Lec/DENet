import argparse
import os
import random
from copy import deepcopy
from pprint import pprint

import torch
import yaml
from imageio import imsave

from model.head import get_architecture
from augmentation.augmentations import Compose, SquareScale
from dataset.transforms import ToTensor, Normalize
from dataset import WrappedVOCSBDSegmentation5i, WrappedCOCOStuff20i
from trainer.utils import center_print, query_rgb, mask_gray, mask_color, upsample, to_cuda

join_path = os.path.join
tensor = torch.tensor
img_size = 375

# remove random flip during inference
transforms = Compose([
    SquareScale(size=img_size),
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225))
])


def get_model(inference_config, checkpoint_dir=None):
    center_print('Inference configurations begins')
    pprint(inference_config)
    center_print('Inference configurations ends')

    # model config
    model_config = inference_config['model']
    model_config_without_arch = deepcopy(model_config)
    model_config_without_arch.pop('arch')
    model = get_architecture(model_config['arch'])(**model_config_without_arch)

    # load config
    infer_config = inference_config['inference']
    # for now, just align to train/test config
    checkpoint_root = 'runs'
    if checkpoint_dir is not None:
        checkpoint_root = checkpoint_dir
    load_path = join_path(checkpoint_root, model_config['arch'], infer_config['id'],
                          'best_model.bin' if infer_config['best'] else 'latest_model.bin')

    print('Loading model from: %s' % load_path)
    model.load_state_dict(torch.load(load_path)['model'], strict=False)

    # device config
    device = infer_config['device']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    print('Using device: %d' % device)
    if torch.cuda.is_available() and device is not None:
        print('Moving model to device: %d' % device)
        model = model.cuda()
    elif not torch.cuda.is_available() and device is not None:
        raise ValueError('Trying to use cuda:%d, but torch.cuda.is_available() return `False`' % device)

    model.eval()

    return model


def unsqueeze_zero_dim(*args):
    res = [a.unsqueeze(0) for a in args]
    return res[0] if len(res) == 1 else res


def squeeze_zero_dim(*args):
    res = [a.squeeze(0) for a in args]
    return res[0] if len(res) == 1 else res


def run_DENet_VOC(config):
    ROOT = config['inference']['root']
    roots_path = join_path(ROOT, 'VOC')
    fold = config['inference']['fold']

    result_dir = join_path('results', config['model']['arch'], config['inference']['id'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    CHECKPOINT_DIR = config['inference']['ckpt']
    model = get_model(config, checkpoint_dir=CHECKPOINT_DIR)

    dataset = WrappedVOCSBDSegmentation5i(root=roots_path,
                                          fold=fold,
                                          split='test',
                                          img_size=img_size,
                                          transforms=transforms)

    save_size = config['inference']['save_size']
    cnt = 0
    while cnt < save_size:
        rand_idx = random.randint(0, len(dataset) - 1)
        print('Getting output of DENet from rand_idx: `%d`' % rand_idx)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = dataset[rand_idx]

        cnt += 1
        print('cnt / save_size: ', cnt, ' / ', save_size)
        Is, Ys, Yq, sample_class, Ys_full = tensor(Is), tensor(Ys), tensor(Yq), tensor(sample_class), tensor(Ys_full)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = \
            unsqueeze_zero_dim(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)

        if torch.cuda.is_available():
            Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = to_cuda(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)

        Yq_full_pre, Yq_pre = model(Is, Ys, Iq, sample_class)
        Yq_pre, Yq_full_pre = upsample(Yq_pre, Yq), upsample(Yq_full_pre, Yq_full)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = \
            squeeze_zero_dim(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)
        Yq_pre, Yq_full_pre = squeeze_zero_dim(Yq_pre, Yq_full_pre)

        print('Saving result in: ', result_dir)
        print("sample cls: ", sample_class)
        imsave(join_path(result_dir, 'voc_%d_Is.png' % rand_idx), query_rgb(Is.squeeze()))
        imsave(join_path(result_dir, 'voc_%d_Iq.png' % rand_idx), query_rgb(Iq))
        imsave(join_path(result_dir, 'voc_%d_Ys.png' % rand_idx), mask_gray(Ys.squeeze(), pre=False))
        imsave(join_path(result_dir, 'voc_%d_Yq.png' % rand_idx), mask_gray(Yq, pre=False))
        imsave(join_path(result_dir, 'voc_%d_Ys_full.png' % rand_idx), mask_color(Ys_full.squeeze(), 21, pre=False))
        imsave(join_path(result_dir, 'voc_%d_Yq_full.png' % rand_idx), mask_color(Yq_full, 21, pre=False))
        imsave(join_path(result_dir, 'voc_%d_Yq_pre.png' % rand_idx), mask_gray(Yq_pre, pre=True))
        imsave(join_path(result_dir, 'voc_%d_Yq_full_pre.png' % rand_idx), mask_color(Yq_full_pre, 21, pre=True))


def run_DENet_COCO(config):
    ROOT = config['inference']['root']
    roots_path = join_path(ROOT, 'COCO')
    fold = config['inference']['fold']

    result_dir = join_path('results', config['model']['arch'], config['inference']['id'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    CHECKPOINT_DIR = config['inference']['ckpt']
    model = get_model(config, checkpoint_dir=CHECKPOINT_DIR)

    dataset = WrappedCOCOStuff20i(root=roots_path,
                                  fold=fold,
                                  split='test',
                                  img_size=img_size,
                                  transforms=transforms)

    save_size = config['inference']['save_size']
    cnt = 0
    while cnt < save_size:
        rand_idx = random.randint(0, len(dataset) - 1)
        print('Getting output of DENet from rand_idx: `%d`' % rand_idx)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = dataset[rand_idx]
        Is, Ys, Yq, sample_class, Ys_full = tensor(Is), tensor(Ys), tensor(Yq), tensor(sample_class), tensor(Ys_full)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = \
            unsqueeze_zero_dim(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)

        cnt += 1
        print('cnt / save_size: ', cnt, ' / ', save_size)
        if torch.cuda.is_available():
            Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = to_cuda(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)

        Yq_full_pre, Yq_pre = model(Is, Ys, Iq, sample_class)
        Yq_pre, Yq_full_pre = upsample(Yq_pre, Yq), upsample(Yq_full_pre, Yq_full)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = squeeze_zero_dim(
            Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)
        Yq_pre, Yq_full_pre = squeeze_zero_dim(Yq_pre, Yq_full_pre)

        print('Saving result in: ', result_dir)
        print("sample cls: ", sample_class)
        imsave(join_path(result_dir, 'coco_%d_Is.png' % rand_idx), query_rgb(Is.squeeze()))
        imsave(join_path(result_dir, 'coco_%d_Iq.png' % rand_idx), query_rgb(Iq))
        imsave(join_path(result_dir, 'coco_%d_Ys.png' % rand_idx), mask_gray(Ys.squeeze(), pre=False))
        imsave(join_path(result_dir, 'coco_%d_Yq.png' % rand_idx), mask_gray(Yq, pre=False))
        imsave(join_path(result_dir, 'coco_%d_Ys_full.png' % rand_idx), mask_color(Ys_full.squeeze(), 81, pre=False))
        imsave(join_path(result_dir, 'coco_%d_Yq_full.png' % rand_idx), mask_color(Yq_full, 81, pre=False))
        imsave(join_path(result_dir, 'coco_%d_Yq_pre.png' % rand_idx), mask_gray(Yq_pre, pre=True))
        imsave(join_path(result_dir, 'coco_%d_Yq_full_pre.png' % rand_idx), mask_color(Yq_full_pre, 81, pre=True))


def run_DENet_VOC_2way(config):
    ROOT = config['inference']['root']
    roots_path = join_path(ROOT, 'VOC')
    fold = config['inference']['fold']

    result_dir = join_path('results', config['model']['arch'], config['inference']['id'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    CHECKPOINT_DIR = config['inference']['ckpt']
    model = get_model(config, checkpoint_dir=CHECKPOINT_DIR)

    dataset = WrappedVOCSBDSegmentation5i(root=roots_path,
                                          fold=fold,
                                          split='test',
                                          way=2,
                                          img_size=img_size,
                                          transforms=transforms)

    save_size = config['inference']['save_size']
    cnt = 0
    while cnt < save_size:
        rand_idx = random.randint(0, len(dataset) - 1)
        print('Getting output of DENet from rand_idx: `%d`' % rand_idx)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = dataset[rand_idx]

        cls_1 = sample_class[0]
        cls_2 = sample_class[1]
        Is, Ys, Yq, sample_class, Ys_full = tensor(Is), tensor(Ys), tensor(Yq), tensor(sample_class), tensor(Ys_full)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = \
            unsqueeze_zero_dim(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)
        if torch.cuda.is_available():
            Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = to_cuda(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)

        Yq_full_pre, Yq_pre = model(Is, Ys, Iq, sample_class)

        Is_1 = torch.select(Is, dim=1, index=0)
        Is_2 = torch.select(Is, dim=1, index=1)
        Ys_1 = torch.select(Ys, dim=1, index=0)
        Ys_2 = torch.select(Ys, dim=1, index=1)
        Yq_1 = torch.select(Yq, dim=1, index=0)
        Yq_2 = torch.select(Yq, dim=1, index=1)

        Yq_pre_1 = torch.select(Yq_pre, dim=0, index=0).unsqueeze(0)
        Yq_pre_2 = torch.select(Yq_pre, dim=0, index=1).unsqueeze(0)

        cnt += 1
        print('cnt / save_size: ', cnt, ' / ', save_size)

        Yq_pre_1, Yq_pre_2 = upsample(Yq_pre_1, Yq_1), upsample(Yq_pre_2, Yq_2)
        Yq_full_pre = upsample(Yq_full_pre, Yq_full)

        print("sample cls: ", cls_1, " ", cls_2)
        print('Saving result in: ', result_dir)
        imsave(join_path(result_dir, 'voc_%d_Is_1_2way_cls%d.png' % (rand_idx, cls_1)), query_rgb(Is_1.squeeze()))
        imsave(join_path(result_dir, 'voc_%d_Is_2_2way_cls%d.png' % (rand_idx, cls_2)), query_rgb(Is_2.squeeze()))
        imsave(join_path(result_dir, 'voc_%d_Iq_2way.png' % rand_idx), query_rgb(Iq))
        imsave(join_path(result_dir, 'voc_%d_Ys_1_2way.png' % rand_idx), mask_gray(Ys_1.squeeze(), pre=False))
        imsave(join_path(result_dir, 'voc_%d_Ys_2_2way.png' % rand_idx), mask_gray(Ys_2.squeeze(), pre=False))
        imsave(join_path(result_dir, 'voc_%d_Yq_full.png' % rand_idx), mask_color(Yq_full, 21, pre=False))
        imsave(join_path(result_dir, 'voc_%d_Yq_pre_1_2way.png' % rand_idx),
               mask_gray(Yq_pre_1.squeeze(), pre=True))
        imsave(join_path(result_dir, 'voc_%d_Yq_pre_2_2way.png' % rand_idx),
               mask_gray(Yq_pre_2.squeeze(), pre=True))
        imsave(join_path(result_dir, 'voc_%d_Yq_full_pre.png' % rand_idx),
               mask_color(Yq_full_pre.squeeze(), 21, pre=True))


def run_DENet_COCO_2way(config):
    ROOT = config['inference']['root']
    roots_path = join_path(ROOT, 'COCO')
    fold = config['inference']['fold']

    result_dir = join_path('results', config['model']['arch'], config['inference']['id'])
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    CHECKPOINT_DIR = config['inference']['ckpt']
    model = get_model(config, checkpoint_dir=CHECKPOINT_DIR)

    dataset = WrappedCOCOStuff20i(root=roots_path,
                                  fold=fold,
                                  split='test',
                                  way=2,
                                  img_size=img_size,
                                  transforms=transforms)

    save_size = config['inference']['save_size']
    cnt = 0
    while cnt < save_size:
        rand_idx = random.randint(0, len(dataset) - 1)
        print('Getting output of DENet from rand_idx: `%d`' % rand_idx)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = dataset[rand_idx]

        cls_1 = sample_class[0]
        cls_2 = sample_class[1]
        Is, Ys, Yq, sample_class, Ys_full = tensor(Is), tensor(Ys), tensor(Yq), tensor(sample_class), tensor(Ys_full)
        Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = \
            unsqueeze_zero_dim(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)
        if torch.cuda.is_available():
            Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full = to_cuda(Is, Ys, Iq, Yq, sample_class, Ys_full, Yq_full)

        Yq_full_pre, Yq_pre = model(Is, Ys, Iq, sample_class)

        Is_1 = torch.select(Is, dim=1, index=0)
        Is_2 = torch.select(Is, dim=1, index=1)
        Ys_1 = torch.select(Ys, dim=1, index=0)
        Ys_2 = torch.select(Ys, dim=1, index=1)
        Yq_1 = torch.select(Yq, dim=1, index=0)
        Yq_2 = torch.select(Yq, dim=1, index=1)

        Yq_pre_1 = torch.select(Yq_pre, dim=0, index=0).unsqueeze(0)
        Yq_pre_2 = torch.select(Yq_pre, dim=0, index=1).unsqueeze(0)

        cnt += 1
        print('cnt / save_size: ', cnt, ' / ', save_size)

        Yq_pre_1, Yq_pre_2 = upsample(Yq_pre_1, Yq_1), upsample(Yq_pre_2, Yq_2)
        Yq_full_pre = upsample(Yq_full_pre, Yq_full)

        print("sample cls: ", cls_1, " ", cls_2)
        print('Saving result in: ', result_dir)
        imsave(join_path(result_dir, 'coco_%d_Is_1_2way_cls%d.png' % (rand_idx, cls_1)), query_rgb(Is_1.squeeze()))
        imsave(join_path(result_dir, 'coco_%d_Is_2_2way_cls%d.png' % (rand_idx, cls_2)), query_rgb(Is_2.squeeze()))
        imsave(join_path(result_dir, 'coco_%d_Iq_2way.png' % rand_idx), query_rgb(Iq))
        imsave(join_path(result_dir, 'coco_%d_Ys_1_2way.png' % rand_idx), mask_gray(Ys_1.squeeze(), pre=False))
        imsave(join_path(result_dir, 'coco_%d_Ys_2_2way.png' % rand_idx), mask_gray(Ys_2.squeeze(), pre=False))
        imsave(join_path(result_dir, 'coco_%d_Yq_full.png' % rand_idx), mask_color(Yq_full, 21, pre=False))
        imsave(join_path(result_dir, 'coco_%d_Yq_pre_1_2way.png' % rand_idx),
               mask_gray(Yq_pre_1.squeeze(), pre=True))
        imsave(join_path(result_dir, 'coco_%d_Yq_pre_2_2way.png' % rand_idx),
               mask_gray(Yq_pre_2.squeeze(), pre=True))
        imsave(join_path(result_dir, 'coco_%d_Yq_full_pre.png' % rand_idx),
               mask_color(Yq_full_pre.squeeze(), 21, pre=True))


def arg_parser():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument('--config',
                        type=str,
                        default='config/inference_config.yml',
                        help='Configuration file to use.')

    return parser.parse_args()


if __name__ == '__main__':
    FUNCTION_MAP = {
        'DENet_VOC': run_DENet_VOC,
        'DENet_COCO': run_DENet_COCO,
        'DENet_VOC_2way': run_DENet_VOC_2way,
        'DENet_COCO_2way': run_DENet_COCO_2way,
    }

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    arg = arg_parser()
    config = arg.config

    with open(config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    func = FUNCTION_MAP.get(config['inference']['function'], None)
    if func is None:
        raise ValueError("inference function type '%s' not support currently."
                         % config['inference']['function'])

    func(config)
