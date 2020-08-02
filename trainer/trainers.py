import matplotlib.pyplot as plt
import time
import os
import sys
import torch
import yaml
import random
from pprint import pprint
from copy import deepcopy
from tqdm import tqdm
import math
from os.path import join as join_path
from tensorboardX import SummaryWriter

import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset import get_loader
from model.common import freeze_weights
from model.head import get_architecture
from optimizer import get_optimizer
from scheduler import get_scheduler
from loss import get_loss
from utils import pca
from utils import BinaryIOU as bIOU
from utils import FullIOU as fIOU
from trainer.utils import Timer, AverageMeter, Logger
from trainer.utils import upsample, mask_gray, mask_color, query_rgb, arch2files, process_nway
from trainer.utils import center_print, Name2Metric, get_sorted_dict_str, Name2Meter, to_cuda
from trainer.utils import knowledge_forward_wrapper, knowledge_loss_wrapper
from trainer.utils import pattern_forward_wrapper, pattern_loss_wrapper
from trainer.utils import DENet_forward_wrapper, DENet_loss_wrapper


class _BaseTrainer(object):
    """
    Base trainer to run few-shot semantic segmentation. It supports several
    operations such as `train`, `train_val` and `test`.
    """

    def __init__(self, train_config=None, test_config=None):
        if not (train_config is None and test_config is not None or
                train_config is not None and test_config is None):
            raise ValueError('Only support one of train config or test config.')

        self.device = train_config['train']['device'] if train_config is not None else test_config['test']['device']
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device)
        print('Using device: %d' % self.device)

        if train_config is not None:
            self._check_train_config(train_config)
            self.train_config_file = train_config

            self.model_cfg = self.train_config_file['model']
            self.data_cfg = self.train_config_file['data']
            self.train_cfg = self.train_config_file['train']

            self.train_loader = None
            self.val_loader = None

            self.debug = self.train_cfg.get('debug', False)
            self.debug_and_val = self.train_cfg.get('debug_and_val', False)

            if self.debug:
                center_print('This is debug mode, no tensorboard will be recorded.', repeat_around=3)
                center_print('This is debug mode, no checkpoint will be saved.', repeat_around=3)
                if not self.debug_and_val:
                    center_print('This is debug mode, no validation will be performed.', repeat_around=3)

            if self.train_cfg.get('resume', None) is not None:
                resume = self.train_cfg['resume']
                self.run_id = self.train_cfg['resume']['id']
                self.logdir = os.path.join('runs', self.model_cfg['arch'], self.run_id)
                with open(join_path(self.logdir, 'train_config.yml')) as config_file:
                    self.train_config_file = yaml.load(config_file, Loader=yaml.FullLoader)
                self.model_cfg = self.train_config_file['model']
                self.data_cfg = self.train_config_file['data']
                self.train_cfg = self.train_config_file['train']
                self.train_cfg['resume'] = resume

            else:
                time_format = '%Y-%m-%d...%H.%M.%S'
                default_run_id = time.strftime(time_format, time.localtime(time.time()))
                self.run_id = self.train_cfg.get('id', default_run_id)
                self.logdir = os.path.join('runs', self.model_cfg['arch'], self.run_id)
                if not self.debug:
                    if os.path.exists(self.logdir):
                        raise ValueError("Given id: `%s` is duplicated." % self.run_id)
                    os.makedirs(self.logdir)
                    yaml.dump(self.train_config_file, open(join_path(self.logdir, 'train_config.yml'), 'w'))
                    # Copy the current code of this model to the log dir.
                    model_file = arch2files.get(self.model_cfg['arch'], None)
                    if model_file is not None:
                        target_model_file = os.path.join(self.logdir, "model.py")
                        os.system("cp " + model_file + " " + target_model_file)
            if not self.debug:
                # redirect the std out stream
                sys.stdout = Logger(os.path.join(self.logdir, 'records.txt'))
                print('Run dir: {}'.format(self.logdir))

            center_print('Train configurations begins')
            pprint(self.train_config_file)
            center_print('Train configurations ends')

            self.model_cfg_without_arch = deepcopy(self.model_cfg)
            self.model_cfg_without_arch.pop('arch')
            self.model = get_architecture(self.model_cfg['arch'])(**self.model_cfg_without_arch)
            optimizer_name = self.train_cfg['optimizer']['name']
            self.optimizer_cfg_without_name = deepcopy(self.train_cfg['optimizer'])
            self.optimizer_cfg_without_name.pop('name')
            self.optimizer = get_optimizer(optimizer_name=optimizer_name)(self.model.parameters(),
                                                                          **self.optimizer_cfg_without_name)
            self.loss_func = get_loss(loss_dict=self.train_cfg['loss'])

            self.scheduler = get_scheduler(self.optimizer, scheduler_dict=self.train_cfg.get('scheduler', None))
            self.best_bIOU = 0
            self.best_step = 1
            self.start_epoch = 1
            self.num_epoch = None
            self.shot = self.data_cfg['shot']
            self.num_steps = self.train_cfg['num_steps']
            self.log_steps = self.train_cfg['log_steps']
            self.val_steps = self.train_cfg['val_steps']
            self.testing = False
            self.name2metric = Name2Metric(name=[], metric=[])
            self.name2meter = Name2Meter(name=[], meter=[])

            if self.train_cfg.get('resume', None) is not None:
                self._load_checkpoints(is_best=self.train_cfg['resume'].get('best', False))

        if test_config is not None:
            self._check_test_config(test_config)
            self.test_config = test_config
            self.test_cfg = self.test_config['test']
            self.model_cfg = self.test_config['model']
            self.data_cfg = self.test_config['data']

            self.run_id = self.test_cfg['id']
            self.ckpt = self.test_cfg.get('ckpt', 'runs')
            self.logdir = os.path.join(self.ckpt, self.model_cfg['arch'], self.run_id)
            self.model_cfg_without_arch = deepcopy(self.model_cfg)
            self.model_cfg_without_arch.pop('arch')
            self.model = get_architecture(self.model_cfg['arch'])(**self.model_cfg_without_arch)
            self.shot = self.data_cfg['shot']
            self.way = self.data_cfg['way']
            self.testing = True
            self.test_loader = None

            # redirect the std out stream
            sys.stdout = Logger(os.path.join(self.logdir, 'test-' + str(self.shot) + 'shot.txt'))
            print('Run dir: {}'.format(self.logdir))

            center_print('Test configurations begins')
            pprint(self.test_config)
            center_print('Test configurations ends')

            self._load_checkpoints(is_best=self.test_cfg['best'], is_testing=True)

        if torch.cuda.is_available() and self.device is not None:
            print('Moving model to device: %d' % self.device)
            self.model = self.model.cuda()
        elif not torch.cuda.is_available() and self.device is not None:
            raise ValueError('Trying to use cuda:%d, but torch.cuda.is_available() return `False`' % self.device)

    @staticmethod
    def _check_train_config(train_config):
        raise NotImplementedError

    @staticmethod
    def _check_test_config(test_config):
        raise NotImplementedError

    def _save_checkpoints(self, iter_idx, is_best=False):
        save_dir = join_path(self.logdir, 'best_model.bin' if is_best else 'latest_model.bin')
        torch.save({
            'best_step': self.best_step,
            'epoch': iter_idx,
            'best_bIOU': self.best_bIOU,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, save_dir)

    def _load_checkpoints(self, is_best=False, is_testing=False):
        load_dir = join_path(self.logdir, 'best_model.bin' if is_best else 'latest_model.bin')
        load_dict = torch.load(load_dir)
        self.start_epoch = load_dict['epoch']
        self.best_step = load_dict['best_step']
        self.best_bIOU = load_dict['best_bIOU']
        self.model.load_state_dict(load_dict['model'])
        if not is_testing:
            self.start_epoch = load_dict['epoch']
            self.scheduler.load_state_dict(load_dict['scheduler'])
        print('Load checkpoint from %s, best step %d, best bIOU %.4f' % (load_dir, self.best_step, self.best_bIOU))

    def train(self):
        timer = Timer()
        if not self.debug:
            writer = SummaryWriter(log_dir=self.logdir)
        else:
            writer = None
        center_print('Training process begins')
        for epoch_idx in range(self.start_epoch, self.num_epoch + 1):
            self.name2metric.reset()
            self.name2meter.reset()
            plot_idx = random.randint(1, len(self.train_loader))
            self.optimizer.step()  # In PyTorch 1.1.0 and later,
            self.scheduler.step(epoch=epoch_idx)
            train_loader_gen = tqdm(enumerate(self.train_loader, 1), position=0, leave=True)

            for batch_idx, batch in train_loader_gen:
                step_idx = (epoch_idx - 1) * len(self.train_loader) + batch_idx

                # train step
                self.train_step(data=batch,
                                train_loader_gen=train_loader_gen,
                                epoch_idx=epoch_idx,
                                step_idx=step_idx,
                                batch_idx=batch_idx,
                                writer=writer,
                                plot_idx=plot_idx)

                if step_idx % self.val_steps == 0:
                    print()
                    self.train_val(epoch_idx, step_idx, timer, writer)
                    print()

                if self.num_steps is not None and step_idx == self.num_steps:
                    if writer is not None:
                        writer.close()
                    print()
                    center_print('Training process ends')
                    return

            train_loader_gen.close()
            print()
        writer.close()
        center_print('Training process ends')

    def train_step(self, data,
                   train_loader_gen,
                   epoch_idx, step_idx, batch_idx,
                   writer, plot_idx,
                   **kwargs):
        raise NotImplementedError

    def train_val(self, iter_idx, step_idx, timer, writer):
        raise NotImplementedError

    def fix_randomness(self):
        """Fix randomness when testing the model."""

        freeze_weights(self.model)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def plot_normal_seg_figures(writer, n_steps, num_classes, I, Y, Y_pre, split='train'):
        """
        Plot normal segmentation figures during training process.

        Args:
            writer: the writer object.
            n_steps: current training step.
            num_classes: total number of classes used in the dataset.
            I: image to plot.
            Y: mask to plot.
            Y_pre: predicted mask to plot.
            split: the type of the fold.

        Returns:

        """
        img = plt.figure()
        plt.subplot(1, 3, 1)
        plt.title("I")
        plt.imshow(query_rgb(I))
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title("Y")
        plt.imshow(mask_color(Y, num_classes=num_classes, pre=False))
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title("Y_pre")
        plt.imshow(mask_color(Y_pre, num_classes=num_classes))
        plt.axis('off')
        writer.add_figure("NormalSeg" + '/' + split, img, global_step=n_steps)

    @staticmethod
    def plot_binary_figures(writer, n_steps, Iq, Yq, Yq_pre, cls,
                            Is=None, Ys=None, Ys_pre=None,
                            hidden_Fs=None, hidden_Fq=None, split='val'):
        """
        Plot binary figures during training process.

        Args:
            writer: the writer object.
            n_steps: current training step.
            Iq: query image to plot.
            Yq: query mask to plot.
            Yq_pre: predicted query mask to plot.
            cls: the specified class in the query image.
            Is: support image to plot.
            Ys: support mask to plot.
            Ys_pre: predicted support mask to plot.
            hidden_Fs: support feature map after feature extraction to plot.
            hidden_Fq: query feature map after feature extraction to plot.
            split: the type of the fold.
        """

        def num_not_none(*args):
            return sum([1 if i is not None else 0 for i in args])

        num_support = num_not_none(Is, Ys, Ys_pre, hidden_Fs)
        num_row = 1 if num_support == 0 else 2
        num_col = 3
        if hidden_Fq is not None:
            num_col += 1
        # plot query first
        img = plt.figure()
        plt.subplot(num_row, num_col, 1)
        plt.title("Iq")
        plt.imshow(query_rgb(Iq))
        plt.axis('off')
        plt.subplot(num_row, num_col, 2)
        plt.title("Yq")
        plt.imshow(mask_gray(Yq, pre=False), cmap='gray')
        plt.axis('off')
        plt.subplot(num_row, num_col, 3)
        plt.title("Yq_pre")
        plt.imshow(mask_gray(Yq_pre), cmap='gray')
        plt.axis('off')
        num_has_plotted = 3
        if hidden_Fq is not None:
            num_has_plotted += 1
            plt.subplot(num_row, num_col, num_has_plotted)
            plt.title("hidden_Fq")
            plt.imshow(query_rgb(hidden_Fq))
            plt.axis('off')

        # plot support
        cnt = 0
        if Is is not None:
            cnt += 1
            plt.subplot(num_row, num_col, cnt + num_has_plotted)
            plt.title("Is")
            plt.imshow(query_rgb(Is))
            plt.axis('off')
        if Ys is not None:
            cnt += 1
            plt.subplot(num_row, num_col, cnt + num_has_plotted)
            plt.title("Ys")
            plt.imshow(mask_gray(Ys, pre=False), cmap='gray')
            plt.axis('off')
        if Ys_pre is not None:
            cnt += 1
            plt.subplot(num_row, num_col, cnt + num_has_plotted)
            plt.title("Ys_pre")
            plt.imshow(mask_gray(Ys_pre), cmap='gray')
            plt.axis('off')
        if hidden_Fs is not None:
            cnt += 1
            plt.subplot(num_row, num_col, cnt + num_has_plotted)
            plt.title("hidden_Fs")
            plt.imshow(query_rgb(hidden_Fs))
            plt.axis('off')

        assert cnt == num_support, 'Interval error cnt = %d while num_support = %d' % (cnt, num_support)
        writer.add_figure(str(cls) + '/' + split + '/binary figs', img, global_step=n_steps)

    @staticmethod
    def plot_full_figures(writer, n_steps, num_classes, Iq, Yq_full, Yq_full_pre, cls,
                          Is=None, Ys_full=None, Ys_full_pre=None,
                          split='val'):
        """
        Plot full segmentation figures during training process.

        Args:
            writer: the writer object.
            n_steps: current training step.
            num_classes: total number of classes used in the dataset.
            Iq: query image to plot.
            Yq_full: full query mask to plot.
            Yq_full_pre: full predicted query mask to plot.
            cls: the specified class in the query image.
            Is: support image to plot.
            Ys_full: full support mask to plot.
            Ys_full_pre: full predicted support mask to plot.
            split: the type of the fold.
        """

        def num_not_none(*args):
            return sum([1 if i is not None else 0 for i in args])

        num_support = num_not_none(Is, Ys_full, Ys_full_pre)
        num_row = 1 if num_support == 0 else 2
        # plot query first
        img = plt.figure()
        plt.subplot(num_row, 3, 1)
        plt.title("Iq")
        plt.imshow(query_rgb(Iq))
        plt.axis('off')
        plt.subplot(num_row, 3, 2)
        plt.title("Yq_full")
        plt.imshow(mask_color(Yq_full, num_classes=num_classes, pre=False))
        plt.axis('off')
        plt.subplot(num_row, 3, 3)
        plt.title("Yq_full_pre")
        plt.imshow(mask_color(Yq_full_pre, num_classes=num_classes))
        plt.axis('off')

        # plot support
        cnt = 0
        if Is is not None:
            cnt += 1
            plt.subplot(num_row, 3, cnt + 3)
            plt.title("Is")
            plt.imshow(query_rgb(Is))
            plt.axis('off')
        if Ys_full is not None:
            cnt += 1
            plt.subplot(num_row, 3, cnt + 3)
            plt.title("Ys_full")
            plt.imshow(mask_color(Ys_full, num_classes=num_classes, pre=False))
            plt.axis('off')
        if Ys_full_pre is not None:
            cnt += 1
            plt.subplot(num_row, 3, cnt + 3)
            plt.title("Ys_full_pre")
            plt.imshow(mask_color(Ys_full_pre, num_classes=num_classes))
            plt.axis('off')
        assert cnt == num_support, 'Interval error cnt = %d while num_support = %d' % (cnt, num_support)
        writer.add_figure(str(cls) + '/' + split + '/full figs', img, global_step=n_steps)

    def test(self):
        raise NotImplementedError

    def step(self, data, train=True, **kwargs):
        raise NotImplementedError


class KnowledgeTrainer(_BaseTrainer):
    """
    Knowledge trainer. Use normal segmentation dataset to train the model and test on
    few-shot segmentation dataset.
    Adaptive Masked Proxies for Few-Shot Segmentation (AMP) should use this trainer.
    """

    def __init__(self, train_config=None, test_config=None):
        super(KnowledgeTrainer, self).__init__(train_config, test_config)

        self.max_num_classes = (train_config or test_config)['model']['maximum_num_classes']
        if train_config is not None:
            train_dataset = get_loader(self.data_cfg['train_dataset'])
            data_path = self.data_cfg['path']
            self.batch_size = self.data_cfg['batch_size']
            self.fold = self.data_cfg.get('fold')
            self.train_set = train_dataset(root=data_path,
                                           fold=self.fold,
                                           split='train',
                                           img_size=self.data_cfg['img_size'],
                                           shot=self.shot)

            val_dataset = get_loader(self.data_cfg['val_dataset'])
            self.val_set = val_dataset(root=data_path,
                                       fold=self.fold,
                                       split='val',
                                       img_size=self.data_cfg['img_size'],
                                       shot=self.shot)

            self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.data_cfg['batch_size'],
                                           shuffle=True, num_workers=4)
            self.val_loader = DataLoader(dataset=self.val_set,
                                         batch_size=self.data_cfg['batch_size_val'] or self.data_cfg['batch_size'])

            self.loss_func = knowledge_loss_wrapper(self.loss_func, arch=self.model_cfg['arch'])
            self.name2metric = Name2Metric(name=['bIOU', 'fIOU'], metric=[bIOU, fIOU])
            self.name2meter = Name2Meter(name=['loss'], meter=[AverageMeter])
            self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))

        if test_config is not None:
            dataset = get_loader(self.data_cfg['dataset'])
            self.test_set = dataset(root=self.data_cfg['path'],
                                    img_size=self.data_cfg['img_size'],
                                    split='val',
                                    fold=self.data_cfg['fold'],
                                    shot=self.shot)

            self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.data_cfg['batch_size'])

    @staticmethod
    def _check_train_config(train_config):
        # TODO
        return True

    @staticmethod
    def _check_test_config(test_config):
        # TODO
        return True

    def step(self, data, train=True, **kwargs):
        if train:
            self.model.train()
            Iq, Yq = data[0:2]
            Is = Ys = cls_idx = None
            if torch.cuda.is_available() and self.device is not None:
                (Iq, Yq) = to_cuda(Iq, Yq)
        else:
            self.model.eval()
            Is, Ys, Iq, Yq, cls_idx = data[0:5]
            if torch.cuda.is_available() and self.device is not None:
                (Is, Ys, Iq, Yq, cls_idx) = to_cuda(Is, Ys, Iq, Yq, cls_idx)

        step_idx = kwargs.get('step_idx', None)
        batch_step = kwargs.get('batch_step', None)
        writer = kwargs.get('writer', None)
        plot_idx = kwargs.get('plot_idx', None)
        N = Yq.size(0)

        Yq_pre, Ys_pre = knowledge_forward_wrapper(self.model, self.model_cfg['arch'], Is, Ys, Iq, Yq, cls_idx)
        if not self.testing:
            loss = self.loss_func(Yq_pre, Yq, Ys_pre, Ys)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # plot training
            plot_train = train and step_idx and batch_step and writer and plot_idx
            if plot_train and (batch_step == plot_idx):
                for index in range(N):
                    self.plot_normal_seg_figures(
                        writer=writer, n_steps=step_idx,
                        num_classes=self.max_num_classes,
                        I=Iq[index], Y=Yq[index], Y_pre=Yq_pre[index],
                        split='train' if train else 'val')

            # plot validation
            plot_val = not train and step_idx and batch_step and writer and plot_idx and (batch_step == plot_idx)
            if plot_val:
                cls_idx = cls_idx.data.cpu().numpy()
                for index, cls in enumerate(cls_idx):
                    self.plot_binary_figures(
                        writer=writer, n_steps=step_idx,
                        Iq=Iq[index], Yq=Yq[index], Yq_pre=Yq_pre[index],
                        cls=cls,
                        Is=Is[index].squeeze(), Ys=Ys[index].squeeze(),
                        Ys_pre=Ys_pre[index] if Ys_pre is not None else None,
                        split='train' if train else 'val')

            Yq_pre = upsample(Yq_pre, Yq)
            return Yq_pre, loss.item()
        else:
            Yq_pre = upsample(Yq_pre, Yq)
            return Yq_pre

    def train_step(self, data,
                   train_loader_gen,
                   epoch_idx, step_idx, batch_idx,
                   writer, plot_idx,
                   **kwargs):
        batch = data
        f_iou = self.name2metric['fIOU']
        f_iou.num_classes = self.max_num_classes
        loss_meter = self.name2meter['loss']

        Iq, Yq = data[0:2]
        Yq_pre, train_loss = self.step(batch,
                                       step_idx=step_idx,
                                       batch_step=batch_idx,
                                       writer=writer,
                                       plot_idx=plot_idx)
        f_iou.update(Yq, Yq_pre)
        loss_meter.update(train_loss)

        if step_idx % self.log_steps == 0:
            if writer is not None:
                writer.add_scalar('train/loss', loss_meter.avg, global_step=step_idx)
                writer.add_scalar('train/fIOU', f_iou.mean_iou(), global_step=step_idx)
                writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=step_idx)

        train_loader_gen.set_description(
            'Epoch %d (%2d/%2d), Step %d, Train, Loss %.4f, F-IOU %.4f, LR %.6f' % (
                epoch_idx, batch_idx,
                len(self.train_loader), step_idx,
                loss_meter.avg, f_iou.mean_iou(),
                self.scheduler.get_lr()[0]))
        self.name2metric['fIOU'] = f_iou
        self.name2meter['loss'] = loss_meter

    def train_val(self, iter_idx, step_idx, timer, writer):
        plot_idx = random.randint(1, len(self.val_loader))
        with torch.no_grad():
            if not self.debug or self.debug_and_val:
                b_iou = bIOU()
                loss_meter = AverageMeter()
                val_loader_gen = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
                # data -> (Is, Ys, Iq, Yq, cls_idx, Ys_full, Yq_full)
                for val_j, data in val_loader_gen:
                    Yq, cls_idx = data[3:5]
                    cls_idx = cls_idx.data.cpu().numpy()
                    Yq_pre, val_loss = self.step(data=data,
                                                 train=False,
                                                 step_idx=step_idx,
                                                 batch_step=val_j,
                                                 writer=writer,
                                                 plot_idx=plot_idx)

                    b_iou.update(Yq, Yq_pre, cls_idx)
                    loss_meter.update(val_loss)
                    val_loader_gen.set_description(
                        'Epoch %d (%2d/%2d), Step %d, Eval, Loss %.4f, B-IOU %.4f' % (
                            iter_idx, val_j,
                            len(self.val_loader), step_idx,
                            loss_meter.avg,
                            b_iou.mean_iou()))
                print('Epoch {}, Eval, Class B-IOU: {}'.format(iter_idx, get_sorted_dict_str(b_iou.class_iou())))
                if writer is not None:
                    writer.add_scalar('val/loss', loss_meter.avg, global_step=step_idx)
                    writer.add_scalar('val/bIOU', b_iou.mean_iou(), global_step=step_idx)
                    if b_iou.mean_iou() > self.best_bIOU:
                        self.best_bIOU = b_iou.mean_iou()
                        self.best_step = step_idx
                        self._save_checkpoints(iter_idx, is_best=True)
                    print('Best Step: {}. Best B-IOU: {:.4f}. Running Time: {}. Estimated Time: {}'.format(
                        self.best_step,
                        self.best_bIOU,
                        timer.measure(),
                        timer.measure(step_idx / self.num_steps)))
                    self._save_checkpoints(iter_idx, is_best=False)

    def test(self):
        self.fix_randomness()
        b_iou = bIOU()
        test_loader_gen = tqdm(enumerate(self.test_loader, 1), position=0, leave=True)
        # data -> (Is, Ys, Iq, Yq, cls_idx, Ys_full, Yq_full)
        for j, batch in test_loader_gen:
            Yq = batch[3]
            cls_idx = batch[4].data.cpu().numpy()
            Yq_pre = self.step(data=batch, train=False)

            b_iou.update(Yq, Yq_pre, cls_idx)
            test_loader_gen.set_description(
                'Test, Progress (%2d/%2d), B-IOU %.4f' % (j, len(self.test_loader), b_iou.mean_iou()))

        fold = self.data_cfg['fold']
        shot = self.data_cfg.get('shot', 1)
        print('Fold %d, shot %d, B-IOU: %.4f' % (fold, shot, b_iou.mean_iou()))
        print('Class B-IOU: ', get_sorted_dict_str(b_iou.class_iou()))


class PatternTrainer(_BaseTrainer):
    """
    Pattern trainer. Use binary segmentation masks to do few-shot training.
    CANet and PGNet should use this trainer.
    """

    def __init__(self, train_config=None, test_config=None):
        super(PatternTrainer, self).__init__(train_config, test_config)

        if train_config is not None:
            train_dataset = get_loader(self.data_cfg['train_dataset'])
            data_path = self.data_cfg['path']
            self.train_set = train_dataset(root=data_path,
                                           fold=self.data_cfg.get('fold'),
                                           split='train',
                                           img_size=self.data_cfg['img_size'],
                                           shot=self.shot)

            val_dataset = get_loader(self.data_cfg['val_dataset'])
            self.val_set = val_dataset(root=data_path,
                                       fold=self.data_cfg.get('fold'),
                                       split='val',
                                       img_size=self.data_cfg['img_size'],
                                       shot=self.shot)

            self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.data_cfg['batch_size'],
                                           shuffle=True, num_workers=4)
            self.val_loader = DataLoader(dataset=self.val_set,
                                         batch_size=self.data_cfg['batch_size_val'] or self.data_cfg['batch_size'])

            self.loss_func = pattern_loss_wrapper(self.loss_func, arch=self.model_cfg['arch'])
            self.name2metric = Name2Metric(name=['bIOU'], metric=[bIOU])
            self.name2meter = Name2Meter(name=['loss'], meter=[AverageMeter])
            self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))

        if test_config is not None:
            dataset = get_loader(self.data_cfg['dataset'])
            self.test_set = dataset(root=self.data_cfg['path'],
                                    fold=self.data_cfg['fold'],
                                    split='test',
                                    img_size=self.data_cfg['img_size'],
                                    shot=self.shot,
                                    way=self.way)

            self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.data_cfg['batch_size'])

    @staticmethod
    def _check_train_config(train_config):
        # TODO
        return True

    @staticmethod
    def _check_test_config(test_config):
        # TODO
        return True

    def step(self, data, train=True, **kwargs):
        if train:
            self.model.train()
        else:
            self.model.eval()

        step_idx = kwargs.get('step_idx', None)
        batch_step = kwargs.get('batch_step', None)
        writer = kwargs.get('writer', None)
        plot_idx = kwargs.get('plot_idx', None)

        Is, Ys, Iq, Yq, cls_idx = data[0:5]
        if torch.cuda.is_available() and self.device is not None:
            (Is, Ys, Iq, Yq) = to_cuda(Is, Ys, Iq, Yq)

        Yq_pre, Ys_pre = pattern_forward_wrapper(self.model, self.model_cfg['arch'], Is, Ys, Iq, Yq)
        if not self.testing:
            loss = self.loss_func(Yq_pre, Yq, Ys_pre, Ys)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # plot
            plot = step_idx and batch_step and writer and plot_idx and (batch_step == plot_idx)
            if plot:
                cls_idx = cls_idx.data.cpu().numpy()
                for index, cls in enumerate(cls_idx):
                    self.plot_binary_figures(
                        writer=writer, n_steps=step_idx,
                        Iq=Iq[index], Yq=Yq[index], Yq_pre=Yq_pre[index],
                        cls=cls,
                        Is=Is[index].squeeze(), Ys=Ys[index].squeeze(),
                        Ys_pre=Ys_pre[index] if Ys_pre is not None else None,
                        split='train' if train else 'val')

            Yq_pre = upsample(Yq_pre, Yq)
            return Yq_pre, loss.item()
        else:
            Yq_pre = upsample(Yq_pre, Yq)
            return Yq_pre

    def train_step(self, data,
                   train_loader_gen,
                   epoch_idx, step_idx, batch_idx,
                   writer, plot_idx,
                   **kwargs):
        batch = data
        b_iou = self.name2metric['bIOU']
        loss_meter = self.name2meter['loss']

        Yq, cls_idx = batch[3:5]
        cls_idx = batch[4].data.cpu().numpy()
        Yq_pre, train_loss = self.step(batch,
                                       step_idx=step_idx,
                                       batch_step=batch_idx,
                                       writer=writer,
                                       plot_idx=plot_idx)

        b_iou.update(Yq, Yq_pre, cls_idx)
        loss_meter.update(train_loss)

        if step_idx % self.log_steps == 0:
            if writer is not None:
                writer.add_scalar('train/loss', loss_meter.avg, global_step=step_idx)
                writer.add_scalar('train/bIOU', b_iou.mean_iou(), global_step=step_idx)
                writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=step_idx)

        train_loader_gen.set_description(
            'Epoch %d (%2d/%2d), Step %d, Train, Loss %.4f, B-IOU %.4f, LR %.6f' % (
                epoch_idx, batch_idx,
                len(self.train_loader), step_idx,
                loss_meter.avg, b_iou.mean_iou(),
                self.scheduler.get_lr()[0]))
        self.name2metric['bIOU'] = b_iou
        self.name2meter['loss'] = loss_meter

    def train_val(self, iter_idx, step_idx, timer, writer):
        plot_idx = random.randint(1, len(self.val_loader))
        with torch.no_grad():
            if not self.debug or self.debug_and_val:
                b_iou = bIOU()
                loss_meter = AverageMeter()
                val_loader_gen = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
                # data -> (Is, Ys, Iq, Yq, cls_idx, Ys_full, Yq_full)
                for val_j, data in val_loader_gen:
                    Is, Ys, Iq, Yq, cls_idx = data[:5]
                    cls_idx = cls_idx.data.cpu().numpy()
                    Yq_pre, val_loss = self.step(data=data,
                                                 train=False,
                                                 step_idx=step_idx,
                                                 batch_step=val_j,
                                                 writer=writer,
                                                 plot_idx=plot_idx)

                    b_iou.update(Yq, Yq_pre, cls_idx)
                    loss_meter.update(val_loss)
                    val_loader_gen.set_description(
                        'Epoch %d (%2d/%2d), Step %d, Eval, Loss %.4f, B-IOU %.4f' % (
                            iter_idx, val_j,
                            len(self.val_loader), step_idx,
                            loss_meter.avg,
                            b_iou.mean_iou()))
                print('Epoch {}, Eval, Class B-IOU: {}'.format(iter_idx, get_sorted_dict_str(b_iou.class_iou())))
                if writer is not None:
                    writer.add_scalar('val/loss', loss_meter.avg, global_step=step_idx)
                    writer.add_scalar('val/bIOU', b_iou.mean_iou(), global_step=step_idx)
                    if b_iou.mean_iou() > self.best_bIOU:
                        self.best_bIOU = b_iou.mean_iou()
                        self.best_step = step_idx
                        self._save_checkpoints(iter_idx, is_best=True)
                    print('Best Step: {}. Best B-IOU: {:.4f}. Running Time: {}. Estimated Time: {}'.format(
                        self.best_step,
                        self.best_bIOU,
                        timer.measure(),
                        timer.measure(step_idx / self.num_steps)))
                    self._save_checkpoints(iter_idx, is_best=False)

    def test(self):
        self.fix_randomness()
        b_iou = bIOU()
        test_loader_gen = tqdm(enumerate(self.test_loader, 1), position=0, leave=True)
        for j, batch in test_loader_gen:
            if self.way > 1:
                pre = []
                for _ in range(self.way):
                    # N, 1, shot, c, h, w
                    Is = batch[0].index_select(dim=1, index=torch.tensor(_))
                    # N, 1, shot, h, w
                    Ys = batch[1].index_select(dim=1, index=torch.tensor(_))
                    # N, c, h, w
                    Iq = batch[2]
                    # N, h, w
                    Yq = torch.select(batch[3], dim=1, index=_)
                    # N
                    cls_idx = torch.select(batch[4], dim=1, index=_).data.cpu().numpy()
                    Yq_pre = self.step((Is, Ys, Iq, Yq, cls_idx), train=False)
                    pre.append(Yq_pre)
                # [N, 2*way, h, w]
                pre = torch.cat(pre, dim=1)
                # [N, way, 2, h, w]
                pre = process_nway(pre, self.way)
                for i in range(self.way):
                    way_Yq = torch.select(batch[3], dim=1, index=i)
                    way_Yq_pre = torch.select(pre, dim=1, index=i)
                    class_id = torch.select(batch[4], dim=1, index=i).data.cpu().numpy()
                    b_iou.update(way_Yq, way_Yq_pre, class_id)
            else:
                Yq = batch[3]
                cls_idx = batch[4].data.cpu().numpy()
                Yq_pre = self.step(batch, train=False)

                b_iou.update(Yq, Yq_pre, cls_idx)

            test_loader_gen.set_description(
                'Test, Progress (%2d/%2d), B-IOU %.4f' % (j, len(self.test_loader), b_iou.mean_iou()))

        fold = self.data_cfg['fold']
        shot = self.data_cfg.get('shot', 1)
        print('Fold %d, shot %d, B-IOU: %.4f' % (fold, shot, b_iou.mean_iou()))
        print('Class B-IOU: ', get_sorted_dict_str(b_iou.class_iou()))


class DENetTrainer(_BaseTrainer):
    """
    DENet trainer which accommodates to the dynamic extension feature of this network.
    DENet should use this trainer.
    """

    @staticmethod
    def _check_train_config(train_config):
        # TODO
        return True

    @staticmethod
    def _check_test_config(test_config):
        # TODO
        return True

    def __init__(self, train_config=None, test_config=None):
        super().__init__(train_config, test_config)

        self.max_num_classes = (train_config or test_config)['model']['maximum_num_classes']
        if train_config is not None:
            train_dataset = get_loader(self.data_cfg['train_dataset'])
            data_path = self.data_cfg['path']
            self.batch_size = self.data_cfg['batch_size']
            self.train_set = train_dataset(root=data_path,
                                           fold=self.data_cfg.get('fold'),
                                           split='train',
                                           img_size=self.data_cfg['img_size'],
                                           shot=self.shot)

            val_dataset = get_loader(self.data_cfg['val_dataset'])
            self.val_set = val_dataset(root=data_path,
                                       fold=self.data_cfg.get('fold'),
                                       split='val',
                                       img_size=self.data_cfg['img_size'],
                                       shot=self.shot)

            self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.data_cfg['batch_size'],
                                           shuffle=True, num_workers=4)
            self.val_loader = DataLoader(dataset=self.val_set,
                                         batch_size=self.data_cfg['batch_size_val'] or self.data_cfg['batch_size'])

            self.loss_func = DENet_loss_wrapper(self.loss_func, arch=self.model_cfg['arch'])
            self.name2metric = Name2Metric(name=['bIOU', 'fIOU'], metric=[bIOU, fIOU])
            self.name2meter = Name2Meter(name=['loss', 'loss_k', 'loss_p'],
                                         meter=[AverageMeter, AverageMeter, AverageMeter])
            self.num_epoch = math.ceil(self.num_steps / len(self.train_loader))
            self.lambda_ = self.train_cfg.get("lambda", 1.0)

        if test_config is not None:
            dataset = get_loader(self.data_cfg['dataset'])
            self.test_set = dataset(root=self.data_cfg['path'],
                                    fold=self.data_cfg['fold'],
                                    split='test',
                                    img_size=self.data_cfg['img_size'],
                                    shot=self.shot,
                                    way=self.way)

            self.test_loader = DataLoader(dataset=self.test_set, batch_size=self.data_cfg['batch_size'])

    def step(self, data, train=True, **kwargs):
        if train:
            self.model.train()
        else:
            self.model.eval()

        step_idx = kwargs.get('step_idx', None)
        batch_step = kwargs.get('batch_step', None)
        writer = kwargs.get('writer', None)
        plot_idx = kwargs.get('plot_idx', None)

        Is, Ys, Iq, Yq, cls_idx, Ys_full, Yq_full = data
        if torch.cuda.is_available() and self.device is not None:
            (Is, Ys, Iq, Yq, cls_idx, Ys_full, Yq_full) = to_cuda(Is, Ys, Iq, Yq, cls_idx, Ys_full, Yq_full)

        Yq_pre, Ys_pre, Yq_full_pre, Ys_full_pre = DENet_forward_wrapper(self.model, self.model_cfg['arch'],
                                                                         Is, Ys, Iq, Yq, cls_idx,
                                                                         Ys_full, Yq_full)
        Yq_pre = Yq_pre.float()
        Yq_pre = F.interpolate(Yq_pre, size=Yq.size()[-2:], mode='bilinear', align_corners=True)
        Yq_full_pre = F.interpolate(Yq_full_pre, size=Yq.size()[-2:], mode='bilinear', align_corners=True)

        if not self.testing:
            # [N, c]
            d = Yq_full_pre.device
            loss_weighted = torch.ones(Yq_full_pre.shape[:2]).cuda(d)
            # re-weight the loss, only support 1-way currently
            loss_weighted[torch.arange(loss_weighted.shape[0]), cls_idx] = self.lambda_

            loss_k, loss_p = self.loss_func(
                Yq_pre, Yq, Ys_pre, Ys, Yq_full_pre, Yq_full,
                Ys_full_pre, Ys_full, loss_weighted)
            loss = loss_k + loss_p

            if train:
                self.optimizer.zero_grad()
                # note that we only backward the loss computed with full masks
                # and full logits.
                loss_k.backward()
                self.optimizer.step()

            # plot figs
            plot = step_idx and batch_step and writer and plot_idx and (batch_step == plot_idx)
            if plot:
                cls_idx = cls_idx.data.cpu().numpy()
                for index, cls in enumerate(cls_idx):
                    Hs = self.model.vis.get("hidden_Fs", None)
                    if Hs is not None:
                        Hs = pca(Hs)
                        Hs = F.interpolate(Hs, size=Yq.size()[-2:], mode='bilinear', align_corners=True)
                    Hq = self.model.vis.get("hidden_Fq", None)
                    if Hq is not None:
                        Hq = pca(Hq)
                        Hq = F.interpolate(Hq, size=Yq.size()[-2:], mode='bilinear', align_corners=True)
                    # plot binary iou figure
                    self.plot_binary_figures(writer, step_idx,
                                             Iq[index], Yq[index], Yq_pre[index], cls,
                                             Is=Is[index][0], Ys=Ys[index][0],
                                             hidden_Fs=Hs[index] if Hs is not None else None,
                                             hidden_Fq=Hq[index] if Hq is not None else None,
                                             split='train' if train else 'val')
                    # plot full iou figure
                    self.plot_full_figures(writer, step_idx, self.max_num_classes,
                                           Iq[index], Yq_full[index], Yq_full_pre[index], cls,
                                           Is=Is[index][0], Ys_full=Ys_full[index][0],
                                           split='train' if train else 'val')

            return Yq_pre, Yq_full_pre, loss.item(), loss_k.item(), loss_p.item()
        else:
            return Yq_pre, Yq_full_pre,

    def train_step(self, data,
                   train_loader_gen,
                   epoch_idx, step_idx, batch_idx,
                   writer, plot_idx,
                   **kwargs):
        batch = data
        b_iou = self.name2metric['bIOU']
        f_iou = self.name2metric['fIOU']
        f_iou.num_classes = self.max_num_classes
        loss_meter = self.name2meter['loss']
        loss_k_meter = self.name2meter['loss_k']
        loss_p_meter = self.name2meter['loss_p']

        Yq_pre, Yq_full_pre, train_loss, loss_k, loss_p = self.step(data=batch,
                                                                    step_idx=step_idx,
                                                                    batch_step=batch_idx,
                                                                    writer=writer,
                                                                    plot_idx=plot_idx)

        Yq, Yq_full = batch[3], batch[6]
        cls_idx = batch[4].data.cpu().numpy()
        b_iou.update(Yq, Yq_pre, cls_idx)
        f_iou.update(Yq_full, Yq_full_pre)

        loss_meter.update(train_loss)
        loss_k_meter.update(loss_k)
        loss_p_meter.update(loss_p)

        if step_idx % self.log_steps == 0:
            if writer is not None:
                writer.add_scalar('train/loss', loss_meter.avg, global_step=step_idx)
                writer.add_scalar('train/loss_k', loss_k_meter.avg, global_step=step_idx)
                writer.add_scalar('train/loss_p', loss_p_meter.avg, global_step=step_idx)
                writer.add_scalar('train/bIOU', b_iou.mean_iou(), global_step=step_idx)
                writer.add_scalar('train/fIOU', f_iou.mean_iou(), global_step=step_idx)
                writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=step_idx)

        train_loader_gen.set_description(
            'Epoch %d (%2d/%2d), Step %d, Train, Loss %.4f, B-IOU %.4f, F-IOU %.4f, LR %.6f' % (
                epoch_idx, batch_idx, len(self.train_loader), step_idx,
                train_loss, b_iou.mean_iou(), f_iou.mean_iou(),
                self.scheduler.get_lr()[0]))
        self.name2metric['bIOU'] = b_iou
        self.name2metric['fIOU'] = f_iou

        self.name2meter['loss'] = loss_meter
        self.name2meter['loss_k'] = loss_k_meter
        self.name2meter['loss_p'] = loss_p_meter

    def train_val(self, iter_idx, step_idx, timer, writer):
        plot_idx = random.randint(1, len(self.val_loader))
        with torch.no_grad():
            if not self.debug or self.debug_and_val:
                val_k_loss_meter = AverageMeter()
                val_p_loss_meter = AverageMeter()
                val_total_loss_meter = AverageMeter()
                b_iou = bIOU()
                f_iou = fIOU(num_classes=self.max_num_classes)
                val_loader_gen = tqdm(enumerate(self.val_loader, 1), position=0, leave=True)
                # data -> (Is, Ys, Iq, Yq, cls_idx, history_mask(optional), index(optional))
                for val_j, data in val_loader_gen:
                    Yq_pre, Yq_full_pre, val_loss, loss_k, loss_p = self.step(data=data,
                                                                              train=False,
                                                                              step_idx=step_idx,
                                                                              batch_step=val_j,
                                                                              writer=writer,
                                                                              plot_idx=plot_idx)
                    val_k_loss_meter.update(loss_k)
                    val_p_loss_meter.update(loss_p)
                    val_total_loss_meter.update(val_loss)

                    Yq, Yq_full = data[3], data[6]
                    cls_idx = data[4].data.cpu().numpy()
                    b_iou.update(Yq, Yq_pre, cls_idx)
                    f_iou.update(Yq_full, Yq_full_pre)
                    val_loader_gen.set_description(
                        'Epoch %d (%2d/%2d), Step %d, Eval, Loss %.4f, B-IOU %.4f, F-IOU %.4f' % (
                            iter_idx, val_j,
                            len(self.val_loader), step_idx,
                            val_total_loss_meter.avg,
                            b_iou.mean_iou(),
                            f_iou.mean_iou()))

                print('Epoch {}, Eval, Class B-IOU: {}'.format(iter_idx, get_sorted_dict_str(b_iou.class_iou())))
                print('Epoch {}, Eval, Class F-IOU: {}'.format(iter_idx, get_sorted_dict_str(f_iou.class_iou())))

                if writer is not None:
                    writer.add_scalar('val/loss', val_total_loss_meter.avg, global_step=step_idx)
                    writer.add_scalar('val/loss_k', val_k_loss_meter.avg, global_step=step_idx)
                    writer.add_scalar('val/loss_p', val_p_loss_meter.avg, global_step=step_idx)
                    writer.add_scalar('val/bIOU', b_iou.mean_iou(), global_step=step_idx)
                    writer.add_scalar('val/fIOU', f_iou.mean_iou(), global_step=step_idx)
                    if b_iou.mean_iou() > self.best_bIOU:
                        self.best_bIOU = b_iou.mean_iou()
                        self.best_step = step_idx
                        self._save_checkpoints(iter_idx, is_best=True)
                    print('Best Step: {}. Best B-IOU: {:.4f}. Running Time: {}. Estimated Time: {}'.format(
                        self.best_step,
                        self.best_bIOU,
                        timer.measure(),
                        timer.measure(step_idx / self.num_steps)))
                    self._save_checkpoints(iter_idx, is_best=False)

    def test(self):
        self.fix_randomness()
        b_iou = bIOU()
        f_iou = fIOU(num_classes=self.max_num_classes)
        test_loader_gen = tqdm(enumerate(self.test_loader, 1), position=0, leave=True)
        for j, batch in test_loader_gen:
            Yq, Yq_full = batch[3], batch[6]
            cls_idx = batch[4]
            Yq_pre, Yq_full_pre = self.step(batch, train=False)
            if self.way > 1:
                # [N, way, 2, h, w]
                Yq_pre = Yq_pre.reshape([Yq.size(0), self.way, *Yq_pre.size()[-3:]])
                for i in range(self.way):
                    way_Yq = torch.select(Yq, dim=1, index=i)
                    way_Yq_pre = torch.select(Yq_pre, dim=1, index=i)
                    class_id = torch.select(cls_idx, dim=1, index=i).data.cpu().numpy()
                    b_iou.update(way_Yq, way_Yq_pre, class_id)
            else:
                cls_idx = cls_idx.data.cpu().numpy()
                b_iou.update(Yq, Yq_pre, cls_idx)
            f_iou.update(Yq_full, Yq_full_pre)
            test_loader_gen.set_description(
                'Test, Progress (%2d/%2d), B-IOU %.4f, F-IOU %.4f' % (j, len(self.test_loader),
                                                                      b_iou.mean_iou(), f_iou.mean_iou()))

        fold = self.data_cfg['fold']
        way = self.way
        shot = self.shot
        print('Fold %d, way %d, shot %d, B-IOU: %.4f, F-IOU: %.4f' % (
            fold, way, shot, b_iou.mean_iou(), f_iou.mean_iou()))
        print('Class B-IOU: ', get_sorted_dict_str(b_iou.class_iou()))
        print('Class F-IOU: ', get_sorted_dict_str(f_iou.class_iou()))


if __name__ == "__main__":
    pass
