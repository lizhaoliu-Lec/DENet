import copy
import os
import pickle
from collections import OrderedDict

import numpy as np
import scipy.misc as misc
from PIL import Image
from torchvision.datasets import SBDataset
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm

from augmentation import SquareScale, RandomHorizontallyFlip, Compose
from dataset.transforms import ToTensor, Normalize
from utils import replace_array_ele_as_dict

# freeze seed for reproducity
np.random.seed(1234)


class VOCSBDSegmentation5(VOCSegmentation):
    """
    VOCSBDSegmentation5 Dataset. Select the 15 classes + background during training,
    and all the 21 classes in test. For more detail how to hold out the 5 classes for each fold,
    please refer to One-Shot Learning for Semantic Segmentation <https://arxiv.org/pdf/1709.03410>.
    We construct it based on VOC2012 and SBD dataset.

    Args:
        roots, root_voc, root_sbd: the root(s) path for saving dataset. Only support roots or
            (root_voc and root_sbd).
        roots: the path where both VOC and SBD are saved in.
        root_voc: the path where VOC is saved in.
        root_sbd: the path where SBD is saved in.
        fold: during training, for fold X, the classes in range(5*X + 1, 5*(X+1) + 1) are hold
            out from training. If None, return to normal settings.
        ignore_idx: integer to fill the ignore places.
        image_set: either `train` or `test` for this dataset.
        download: whether to download the dataset.
        transform, target_transform, transforms: inherit directly from VOCSegmentation.
        rebuild_mask: whether to rebuild the mask of the dataset.
    """
    ID2CLASS_NAME = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'dining table',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'potted plant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tv monitor',
    }

    def __init__(self,
                 roots=None,
                 root_voc=None,
                 root_sbd=None,
                 fold=None,
                 ignore_idx=255,
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 rebuild_mask=False):
        # There is no validation in this dataset.
        if image_set == 'val' or image_set == 'validation':
            image_set = 'test'
        self._check_roots(roots=roots, root_voc=root_voc, root_sdb=root_sbd)
        super().__init__(root=roots or root_voc,
                         year='2012',
                         image_set='train' if image_set == 'train' else 'val',
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         transforms=transforms)
        self.sbd = SBDataset(root=roots or root_sbd,
                             image_set='train_noval' if image_set == 'train' else 'val',
                             mode='segmentation',
                             download=download,
                             transforms=None)
        self.roots = roots or root_voc
        self.image_set = image_set
        self.fold = fold
        self.ignore_idx = ignore_idx
        self.rebuild_mask = rebuild_mask
        self.voc_image_base_dir = os.path.split(self.images[0])[0]
        self.sbd_image_base_dir = os.path.join(self.sbd.root, 'img')
        self.images = self._combine_voc_and_sbd(self.images, self.sbd.images)
        self.masks = self._combine_voc_and_sbd(self.masks, self.sbd.masks)
        self.voc_mask_base_dir, self.sbd_mask_base_dir = self._get_both_mask_base_dirs()
        assert len(self.images) == len(self.masks), 'images len: %d, masks len: %d' % (len(self.images),
                                                                                       len(self.masks))
        if self.fold is not None:
            self._fold()
            assert len(self.images) == len(self.masks), 'images len: %d, masks len: %d' % (len(self.images),
                                                                                           len(self.masks))

    def __getitem__(self, index):
        self.assert_image_mask_consistent(self.images[index], self.masks[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.masks[index].endswith('.mat'):
            target = self.sbd._get_target(self.masks[index])
        else:
            target = Image.open(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def assert_image_mask_consistent(self, image_path, mask_path):
        image_idx = self._get_image_or_mask_id(image_path)
        mask_idx = self._get_image_or_mask_id(mask_path)
        assert image_idx == mask_idx, 'got inconsistent image_path: %s and mask_path: %s' % (image_path,
                                                                                             mask_path)

    def _combine_voc_and_sbd(self, voc, sbd):
        """
        Combine the PASCAL dataset and the SBD dataset.

        Args:
            voc: images or masks of voc.
            sbd: images or masks of sbd.

        Returns:
            combined_images or combined_masks.
        """
        combined = list()
        joint_idx = set()
        for i in voc + sbd:
            idx = self._get_image_or_mask_id(i)
            if idx not in joint_idx:
                joint_idx.add(idx)
                combined.append(i)

        return combined

    @staticmethod
    def _get_image_or_mask_id(image_or_mask):
        return image_or_mask.split('/')[-1].split('.')[0]

    @staticmethod
    def _check_roots(roots, root_voc, root_sdb):
        if roots is not None and (root_voc is not None or root_sdb is not None):
            raise ValueError('when given roots, root_voc and root_sdb must be both None.')
        if roots is None and (root_voc is None or root_sdb is None):
            raise ValueError('when not given roots, root_voc and root_sdb must be both specified.')

    @staticmethod
    def squeeze(Is, Ys, Iq, Yq):
        return Is.squeeze(), Ys.squeeze(), Iq, Yq

    def _get_both_mask_base_dirs(self):
        voc_mask_base_dir = os.path.split([i for i in self.masks if i.split('.')[-1] == 'png'][0])[0]
        sbd_mask_base_dir = os.path.split([i for i in self.masks if i.split('.')[-1] == 'mat'][0])[0]
        return voc_mask_base_dir, sbd_mask_base_dir

    def _get_both_image_base_dirs(self):

        voc_image_base_dir = os.path.split([i for i in self.images if i.split('.')[-1] == 'png'][0])[0]
        sbd_image_base_dir = os.path.split([i for i in self.images if i.split('.')[-1] == 'mat'][0])[0]
        return voc_image_base_dir, sbd_image_base_dir

    def _fold(self):
        # logic is as follow
        # root path refer to the root of both masks if the root path doesn't exist or the root image
        # exists but has nothing under it, we build the dataset otherwise just get the listdir of
        # the root, and rebuild the image path.
        voc_fold_path = os.path.join(self.voc_mask_base_dir, self.image_set, str(self.fold))
        sbd_fold_path = os.path.join(self.sbd_mask_base_dir, self.image_set, str(self.fold))
        if not os.path.exists(voc_fold_path) or not os.path.exists(sbd_fold_path) or self.rebuild_mask:
            os.makedirs(voc_fold_path, exist_ok=True)
            os.makedirs(sbd_fold_path, exist_ok=True)
            fold_images = list()
            fold_masks = list()
            ignore_classes = self.get_classes_to_ignore()
            ignore_classes2ignore_idx = {ic: self.ignore_idx for ic in ignore_classes}
            if self.image_set == 'train':
                print('\nIgnoring classes for fold %d in split %s: ' % (self.fold, self.image_set), ignore_classes)
            else:
                print('\nPreserving all classes for fold %d in split %s' % (self.fold, self.image_set))
            print('Preparing data for fold %d in split %s' % (self.fold, self.image_set))
            for index in tqdm(range(len(self)), position=0, leave=True):
                original_mask_path = self.masks[index]
                if original_mask_path.endswith('.mat'):
                    fold_mask_root = sbd_fold_path
                else:
                    fold_mask_root = voc_fold_path
                fold_mask_path = os.path.join(fold_mask_root,
                                              os.path.basename(original_mask_path).split('.')[0] + '.png')
                if self.masks[index].endswith('.mat'):
                    target = self.sbd._get_target(self.masks[index])
                else:
                    target = Image.open(self.masks[index])
                target = np.array(target, dtype=np.int)
                # set the unknow to background
                target[target == 255] = 0
                # filter out the SBD boundary, set all to background
                target[target > 20] = 0
                if self.image_set == 'train':  # only filter out some classes during training
                    # old style, slow
                    # for ic in ignore_classes:
                    #     target[target == ic] = self.ignore_idx
                    # new style, maybe faster
                    target = replace_array_ele_as_dict(target, ignore_classes2ignore_idx)
                    # Images with only background and ignored will not be used
                    if target[target != self.ignore_idx].sum() == 0:
                        continue
                if not os.path.exists(fold_mask_path) or self.rebuild_mask:
                    target = misc.toimage(target, low=target.min(), high=target.max())
                    misc.imsave(fold_mask_path, target)
                fold_images.append(self.images[index])
                fold_masks.append(fold_mask_path)

            self.images = fold_images
            self.masks = fold_masks
        else:
            voc_mask_path = [os.path.join(voc_fold_path, i) for i in os.listdir(voc_fold_path)]
            sbd_mask_path = [os.path.join(sbd_fold_path, i) for i in os.listdir(sbd_fold_path)]
            self.masks = voc_mask_path + sbd_mask_path
            new_images = []
            for m_path in voc_mask_path:
                image_path = os.path.join(self.voc_image_base_dir, os.path.basename(m_path).split('.')[0] + '.jpg')
                assert os.path.exists(m_path), 'mask %s not exist' % m_path
                assert os.path.exists(image_path), 'image %s not exist' % image_path
                new_images.append(image_path)
            for m_path in sbd_mask_path:
                image_path = os.path.join(self.sbd_image_base_dir, os.path.basename(m_path).split('.')[0] + '.jpg')
                assert os.path.exists(m_path), 'mask %s not exist' % m_path
                assert os.path.exists(image_path), 'image %s not exist' % image_path
                new_images.append(image_path)
            self.images = new_images

    def get_classes_to_ignore(self):
        ignore_when_train = [i for i in range(self.fold * 5 + 1, (self.fold + 1) * 5 + 1)]
        return ignore_when_train


class VOCSBDSegmentation5i(VOCSBDSegmentation5):
    """
    VOCSBDSegmentation5i Segmentation Dataset. Comparing with VOCSBDSegmentation5, this dataset
    also return the binary mask according to the query class in a pair style.

    Args:
        train_size: due to the fact that we are randomly sample the pair to construct the final
            dataset, we can sample up to O(n^2) image to train the model, which makes it into
            1~million pairs and may not be feasible if you don't need that, so just specify the
            train_size. We sample the classes in a balanced way, the final classes may has a
            smaller number of images & masks.
        test_size: analogy to train_size, but specify for test.
        rebuild_pair: whether to rebuild query support pair.
    """

    def __init__(self,
                 train_size=15000,
                 test_size=1000,
                 rebuild_pair=False,
                 rebuild_whole_pair=False,
                 shot=1,
                 way=1,
                 *args, **kwargs):
        self.support_query = None
        self.train_size = train_size
        self.test_size = test_size
        self.threshold = 0.02
        self.rebuild_pair = rebuild_pair
        self.rebuild_whole_pair = rebuild_whole_pair
        self.shot = shot
        self.way = way
        super().__init__(*args, **kwargs)
        self.ignore_classes = self.get_classes_to_ignore()
        if self.way <= 0 or self.way > len(self.ignore_classes):
            raise ValueError("way should be in the range of [1, %d], but found %d." % (len(self.ignore_classes), way))
        if self.image_set == 'train' and self.way > 1:
            raise ValueError("n-way should only be set when testing, but now is used in training.")
        self.item_class = self.item_with_class()
        self.whole_item_class = self.whole_item_with_class()
        self.support_query = self.construct_support_query()

    def whole_item_with_class(self):
        # only used when testing
        if self.image_set == 'train':
            return None

        file_name = os.path.basename
        item_class_dir = os.path.join(self.roots, 'item_with_class', self.image_set, str(self.fold))
        if not os.path.exists(item_class_dir):
            os.makedirs(item_class_dir)
        whole_pair = 'whole_item_class.pair'
        whole_pair_path = os.path.join(item_class_dir, whole_pair)
        if not os.path.exists(whole_pair_path) or self.rebuild_whole_pair:
            item_class = dict()
            print('Caching whole item class pair to %s' % whole_pair_path)
            for index in tqdm(range(super().__len__()), position=0, leave=True):
                image_path, mask_path = self.images[index], self.masks[index]
                _, mask = super().__getitem__(index)
                mask = np.array(mask, dtype=np.int)
                uniques = np.unique(mask).tolist()
                for c in uniques:
                    # make sure we only include the novel classes
                    if c in self.ignore_classes and c != 0:
                        if c not in item_class.keys():
                            item_class.update({c: [(file_name(image_path), file_name(mask_path))]})
                        else:
                            cls_list = item_class.get(c)
                            cls_list.append((file_name(image_path), file_name(mask_path)))
            for k in item_class.keys():
                np.random.shuffle(item_class[k])
            with open(whole_pair_path, mode='wb') as f:
                pickle.dump(item_class, file=f)

        with open(whole_pair_path, mode='rb') as f:
            item_class = pickle.loads(f.read())
        return item_class

    def item_with_class(self):
        file_name = os.path.basename
        ignore_classes = self.ignore_classes

        def get_mask_fraction_has_at_least_threshold(mask, threshold, ignore_idx):
            m = mask.flatten()
            from collections import Counter
            id2cnt = Counter(m)
            total = sum(id2cnt.values())
            id2fra = {i: c / total for i, c in id2cnt.items()}
            id2fra = {i: f for i, f in id2fra.items() if f >= threshold}
            # always preserve ignore and background
            return list(id2fra.keys()) + [ignore_idx, 0]

        item_class_dir = os.path.join(self.roots, 'item_with_class', self.image_set, str(self.fold),
                                      str(self.threshold))
        if not os.path.exists(item_class_dir):
            os.makedirs(item_class_dir)
        item_class_pair = 'item_class.pair'
        if self.image_set == 'test' and self.way > 1:
            item_class_pair = 'unlimited_item_class.pair'
        item_class_path = os.path.join(item_class_dir, item_class_pair)
        if not os.path.exists(item_class_path) or self.rebuild_pair:
            item_class = list()
            print('Caching item class pair to %s' % item_class_path)
            for index in tqdm(range(super().__len__()), position=0, leave=True):
                image_path, mask_path = self.images[index], self.masks[index]
                _, mask = super().__getitem__(index)
                mask = np.array(mask, dtype=np.int)
                # uniques = np.unique(mask).tolist()
                uniques = get_mask_fraction_has_at_least_threshold(
                    mask, self.threshold, self.ignore_idx)
                if self.image_set == 'train':
                    # depart the root dir considering the portability
                    item_class.extend([(file_name(image_path), file_name(mask_path), c) for c in uniques if
                                       c != self.ignore_idx and c != 0])
                elif self.image_set == 'test' and self.way == 1:
                    # we only select the images that have base classes and
                    # only 1 novel class when testing on 1-way setting.
                    old_length = len(set(ignore_classes))
                    if len(set(ignore_classes) - set(uniques)) != old_length - 1:
                        continue
                    # depart the root dir considering the portability
                    item_class.extend([(file_name(image_path), file_name(mask_path), c) for c in uniques if
                                       c in ignore_classes])
                else:
                    # i.e., image_set == 'test' and way > 1
                    # we relieve the limitation when testing on n-way setting.
                    item_class.extend([(file_name(image_path), file_name(mask_path), c) for c in uniques if
                                       c in ignore_classes])
            np.random.shuffle(item_class)
            with open(item_class_path, mode='wb') as f:
                pickle.dump(item_class, file=f)

        with open(item_class_path, mode='rb') as f:
            item_class = pickle.loads(f.read())
        return item_class

    def construct_support_query(self):
        ignore_classes = self.ignore_classes
        unique_cls = sorted(list(set([item[-1] for item in self.item_class])))
        size = self.test_size if self.image_set == 'test' else self.train_size
        if size % len(unique_cls) == 0:
            num_per_class = size // len(unique_cls)
            num_last_class = num_per_class
        else:
            num_per_class = size // len(unique_cls)
            num_last_class = num_per_class + size - len(unique_cls) * num_per_class

        cls2items = OrderedDict()
        for cls in unique_cls:
            items = []
            for image, mask, cls_id in self.item_class:
                if cls_id == cls:
                    items.append([image, mask])
            cls2items[cls] = items

        support_query = []
        for idx, cls in enumerate(unique_cls):
            items = cls2items[cls]
            num = num_per_class if idx != len(unique_cls) - 1 else num_last_class
            # some classes may have very small number of samples
            # such as 1, then no pair can be constructed
            # for the classes that n^2 < num, there is a lot of duplication
            # we ignore it, however during test, we need to return every classes
            if len(items) ** 2 < num and self.image_set != 'test':
                continue
            cnt = 0
            num_item = len(items)
            while cnt < num:
                randoms_s = np.random.randint(0, num_item, size=self.shot)
                random_q = np.random.randint(0, num_item)
                # assure items in the pair do not duplicated
                total = len({*randoms_s, random_q})
                cls_id = cls
                if total == self.shot + 1:
                    images_s = [items[i][0] for i in randoms_s]
                    masks_s = [items[i][1] for i in randoms_s]

                    if 1 < self.way <= len(ignore_classes):
                        # deep copy the ignore classes
                        all_other_classes = [_ for _ in ignore_classes]
                        # remove the current class id
                        all_other_classes.remove(cls)
                        # sample some classes without replacement
                        other_classes = np.random.choice(all_other_classes, self.way - 1, replace=False)
                        cls_id = [cls, *other_classes]

                        # validate query image contains at most the classes in support set
                        while True:
                            # the classes that should be exclude in query image
                            exclude_classes = set(ignore_classes) - set(cls_id)
                            validate = [tuple(items[random_q]) in self.whole_item_class[i] for i in exclude_classes]
                            # assure items in the pair do not duplicated
                            total = len({*randoms_s, random_q})
                            if True not in validate and total == self.shot + 1:
                                break
                            # some conditions are not satisfied, then re-sample a query image
                            random_q = np.random.randint(0, num_item)

                        for oc in other_classes:
                            oc_items = cls2items[oc]
                            oc_num_item = len(oc_items)
                            # assure items in the support pair do not duplicated
                            while True:
                                oc_randoms_s = np.random.randint(0, oc_num_item, size=self.shot)
                                oc_total = len({*oc_randoms_s})
                                if oc_total == self.shot:
                                    break
                            images_s.extend([oc_items[i][0] for i in oc_randoms_s])
                            masks_s.extend([oc_items[i][1] for i in oc_randoms_s])
                    # when way > 1, cls_id is a list,
                    # when way ==1, cls_id is an integer for compatibility.
                    support_query.append([images_s, masks_s, *items[random_q], cls_id])
                    cnt += 1
        np.random.shuffle(support_query)

        # add base dir
        old_support_query = copy.deepcopy(support_query)
        support_query = []
        for _ in range(len(old_support_query)):
            image_s, mask_s, image_q, mask_q, cls = old_support_query[_]
            images_s, masks_s = [], []
            for img, mask in zip(image_s, mask_s):
                i, m = self._add_root_dir(img, mask)
                images_s.append(i)
                masks_s.append(m)
            image_q, mask_q = self._add_root_dir(image_q, mask_q)
            support_query.append([images_s, masks_s, image_q, mask_q, cls])
        return support_query

    def _add_root_dir(self, img_path, mask_path):
        # first find the mask root, then decide to use voc or sbd
        join = os.path.join
        voc_fold_path = join(self.voc_mask_base_dir, self.image_set, str(self.fold))
        sbd_fold_path = join(self.sbd_mask_base_dir, self.image_set, str(self.fold))
        voc_mask_path = join(voc_fold_path, mask_path)
        sbd_mask_path = join(sbd_fold_path, mask_path)
        if os.path.exists(voc_mask_path):
            mask_path = voc_mask_path
            img_path = join(self.voc_image_base_dir, img_path)
        else:
            mask_path = sbd_mask_path
            img_path = join(self.sbd_image_base_dir, img_path)

        return img_path, mask_path

    def __len__(self):
        if self.support_query is not None:
            return len(self.support_query)
        else:
            return super().__len__()

    def __getitem__(self, item):
        image_s, mask_s, image_q, mask_q, cls = self.support_query[item]
        images_s, masks_s, shot_masks_s = [], [], []
        cnt = 0
        for img, mask in zip(image_s, mask_s):
            cls_id = cls[cnt // self.shot] if isinstance(cls, list) else cls
            i = Image.open(img).convert('RGB')
            m = Image.open(mask)
            if self.transforms is not None:
                i, m = self.transforms(i, m)
            shot_m = copy.deepcopy(m)
            shot_m = np.array(shot_m)
            shot_m[shot_m != cls_id] = 0
            shot_m[shot_m == cls_id] = 1
            images_s.append(i)
            masks_s.append(m)
            shot_masks_s.append(shot_m)
            cnt += 1
        image_q = Image.open(image_q).convert('RGB')
        mask_q = Image.open(mask_q)
        if self.transforms is not None:
            image_q, mask_q = self.transforms(image_q, mask_q)
        if self.way > 1:
            query_classes = np.unique(mask_q).tolist()
            # obtain all the test classes
            test_classes = set(query_classes).intersection(set(self.ignore_classes))
            num_classes = len(test_classes)
            assert 1 <= num_classes <= self.way, \
                "test classes in query image should be in range of [1, way], but found %d." % num_classes
            shot_masks_q = [np.array(copy.deepcopy(mask_q)) for _ in range(self.way)]
            # assure the class order corresponds to the list -- cls.
            for test, shot_mq in zip(cls, shot_masks_q):
                shot_mq[shot_mq != test] = 0
                shot_mq[shot_mq == test] = 1
        else:
            # i.e., way == 1
            shot_masks_q = copy.deepcopy(mask_q)
            shot_masks_q = np.array(shot_masks_q)
            shot_masks_q[shot_masks_q != cls] = 0
            shot_masks_q[shot_masks_q == cls] = 1

        images_s = np.stack(images_s, axis=0)
        shot_masks_s = np.stack(shot_masks_s, axis=0)
        masks_s = np.stack(masks_s, axis=0)
        if self.way > 1:
            cls = np.stack(cls, axis=0)  # [way]
            shot_masks_q = np.stack(shot_masks_q, axis=0)  # [way, h, w]
        images_s = np.reshape(images_s, (self.way, self.shot, *images_s.shape[1:]))
        shot_masks_s = np.reshape(shot_masks_s, (self.way, self.shot, *shot_masks_s.shape[1:]))
        masks_s = np.reshape(masks_s, (self.way, self.shot, *masks_s.shape[1:]))
        return images_s, shot_masks_s, image_q, shot_masks_q, cls, masks_s, mask_q


class WrappedVOCSBDSegmentation5(VOCSBDSegmentation5i):
    """
    WrappedVOCSBDSegmentation5 dataset for normal segmentation task.

    Args:
        split: same as image_set, but with different name.
        img_size: expected img_size for both image and mask.
   """

    def __init__(self, root, fold, split, img_size, *args, **kwargs):
        transforms = Compose([
            SquareScale(size=img_size),
            RandomHorizontallyFlip(p=0.5),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))
        ])
        super().__init__(roots=root,
                         fold=fold,
                         image_set=split,
                         transforms=transforms,
                         *args, **kwargs)
        self.support_query = None
        self.query = self.construct_query()

    def construct_query(self):
        unique_cls = sorted(list(set([item[-1] for item in self.item_class])))
        size = self.test_size if self.image_set == 'test' else self.train_size
        if size % len(unique_cls) == 0:
            num_per_class = size // len(unique_cls)
            num_last_class = num_per_class
        else:
            num_per_class = size // len(unique_cls)
            num_last_class = num_per_class + size - len(unique_cls) * num_per_class

        cls2items = OrderedDict()
        for cls in unique_cls:
            items = []
            for image, mask, cls_id in self.item_class:
                if cls_id == cls:
                    items.append([image, mask])
            cls2items[cls] = items
        done = 0
        query = []
        for idx, cls in enumerate(unique_cls):
            items = cls2items[cls]
            num = num_per_class if idx != len(unique_cls) - 1 else num_last_class
            cnt = 0
            num_item = len(items)
            while cnt < num:
                index = np.random.randint(0, num_item)
                query.append([*items[index], cls])
                cnt += 1
            done += 1
        np.random.shuffle(query)

        # add base dir
        old_query = copy.deepcopy(query)
        query = []
        for _ in range(len(old_query)):
            image, mask, _ = old_query[_]
            image, mask = self._add_root_dir(image, mask)
            query.append([image, mask])
        return query

    def __len__(self):
        if self.query is not None:
            return len(self.query)
        else:
            return super().__len__()

    def __getitem__(self, item):
        image, mask = self.query[item]
        image = Image.open(image).convert('RGB')
        mask = Image.open(mask)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask

    @staticmethod
    def to_cuda(*args):
        return [a.cuda() for a in args]


class WrappedVOCSBDSegmentation5i(VOCSBDSegmentation5i):
    """
    WrappedVOCSBDSegmentation5i dataset for easy out of the box.

    Args:
        split: same as image_set, but with different name.
        img_size: expected img_size for both image and mask.
    """

    def __init__(self, root, fold, split, img_size, *args, **kwargs):
        transforms = Compose([
            SquareScale(size=img_size),
            RandomHorizontallyFlip(p=0.5),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225))
        ])
        super().__init__(roots=root,
                         fold=fold,
                         image_set=split,
                         transforms=transforms,
                         *args, **kwargs)

    @staticmethod
    def to_cuda(*args):
        return [a.cuda() for a in args]


if __name__ == '__main__':
    #################################################################
    # The following functions are just some basic test functions
    # to see if the codes work. But to avoid the annoying
    # Pycharm auto test triggered, we use the `run` instead of `test`.
    ##################################################################

    # specify root path for the data path
    ROOT = '/path/to/data'
    roots_path = os.path.join(ROOT, 'VOC')


    def run_VOCSBDSegmentation5i_from_scratch():
        fold = 0
        voc = VOCSBDSegmentation5i(roots=roots_path,
                                   image_set='train',
                                   fold=fold,
                                   download=True)
        print('split: train, fold: %d, len: ' % fold, len(voc))

        voc = VOCSBDSegmentation5i(roots=roots_path,
                                   image_set='test',
                                   fold=fold,
                                   download=False)  # already download above
        print('split: test, fold: %d, len: ' % fold, len(voc))


    def run_VOCSBDSegmentation5i_wrapper():
        dataset = WrappedVOCSBDSegmentation5i(root=roots_path,
                                              fold=1,
                                              split='train',
                                              img_size=224)
        # test sbd
        idxes = [_ for _ in range(0, len(dataset))]
        import random
        random.shuffle(idxes)

        for i in idxes[:10]:
            image_s, shot_mask_s, image_q, shot_mask_q, cls, mask_s, mask_q = dataset[i]
            print(image_s.shape, shot_mask_s.shape, image_q.shape, shot_mask_q.shape,
                  cls, mask_s.shape, mask_q.shape)


    def run_WrappedVOCSBDSegmentation5i_dataloader():
        from torch.utils import data
        fold = 1
        split = 'train'
        # split = 'test'
        dataset = WrappedVOCSBDSegmentation5i(root=roots_path,
                                              fold=fold,
                                              split=split,
                                              way=1,
                                              img_size=224)

        train_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

        for d in train_loader:
            image_s, shot_mask_s, image_q, shot_mask_q, cls, mask_s, mask_q = d
            print(image_s.shape, shot_mask_s.shape, image_q.shape, shot_mask_q.shape,
                  cls.shape, mask_s.shape, mask_q.shape)


    def run_WrappedVOCSBDSegmentation5_dataloader():
        from torch.utils import data
        dataset = WrappedVOCSBDSegmentation5(root=roots_path,
                                             fold=1,
                                             split='train',
                                             img_size=224)
        train_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

        for d in train_loader:
            imgs, masks = d
            print(imgs.shape, masks.shape)


    def run_WrappedVOCSBDSegmentation5i_network():
        from torch.utils import data
        from model.head.pgn import PGN
        import torch.nn.functional as F
        from loss import cross_entropy2d
        from optimizer import get_optimizer

        batch_size = 4
        epoch = 1

        train_set = WrappedVOCSBDSegmentation5i(root=roots_path,
                                                fold=1,
                                                # remember to run both train and test set
                                                split='test',
                                                rebuild_mask=False,
                                                img_size=224)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

        model = PGN()

        optim = get_optimizer()(model.parameters(), lr=0.0025, momentum=0.9,
                                dampening=0,
                                weight_decay=0, nesterov=False)

        for e in range(epoch):
            for i_iter, data in enumerate(train_loader):
                Is, Ys, Iq, Yq, sample_class, _, _ = data
                Ys, Yq = Ys.unsqueeze(1).float(), Yq.unsqueeze(1).float()

                pred = model(Is, Ys, Iq)

                pred = F.interpolate(pred, size=Yq.size()[-2:], mode='bilinear',
                                     align_corners=True)

                loss = cross_entropy2d(pred, Yq.long())
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(loss.item(), sample_class)


    def run_WrappedVOCSBDSegmentation5_network():
        from torch.utils import data
        from model.head.amp import AMP
        import torch.nn.functional as F
        from loss import cross_entropy2d
        from optimizer import get_optimizer

        batch_size = 4
        epoch = 1

        train_set = WrappedVOCSBDSegmentation5(root=roots_path,
                                               fold=1,
                                               split='train',
                                               rebuild_mask=False,
                                               img_size=224)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

        model = AMP(maximum_num_classes=21)

        optim = get_optimizer()(model.parameters(), lr=0.0025, momentum=0.9,
                                dampening=0,
                                weight_decay=0, nesterov=False)

        for e in range(epoch):
            for i_iter, data in enumerate(train_loader):
                I, Y = data
                Y = Y.unsqueeze(1).float()

                pred = model(I, phase='train')
                pred = F.interpolate(pred, size=Y.size()[-2:], mode='bilinear',
                                     align_corners=True)
                loss = cross_entropy2d(pred, Y.long())
                optim.zero_grad()
                loss.backward()
                optim.step()
                print(loss.item())


    def run_VOCSBDSegmentation5i_rebuild_pair():
        fold = 0
        image_set = 'test'
        VOCSBDSegmentation5i(roots=roots_path,
                             fold=fold,
                             image_set=image_set,
                             rebuild_pair=True,
                             rebuild_mask=True)


    def run_VOCSBDSegmentation5i_2_way_images():
        fold = 1
        dataset = WrappedVOCSBDSegmentation5i(root=roots_path,
                                              fold=fold,
                                              way=2,
                                              split='test',
                                              img_size=224)
        # test sbd
        idxes = [_ for _ in range(0, len(dataset))]
        import random
        random.shuffle(idxes)

        import matplotlib.pyplot as plt
        # n-way data shape (n>1)
        # image_s: (way, shot, 3, h, w)
        # shot_mask_s: (way, shot, h, w)
        # cls: list, length: way
        for i in [0, 200, 400, 600, 800]:
            image_s, shot_mask_s, image_q, shot_mask_q, cls, mask_s, mask_q = dataset[i]
            print(image_s.shape, shot_mask_s.shape, image_q.shape, shot_mask_q.shape,
                  cls, mask_s.shape, mask_q.shape)

            img1 = image_s[0].squeeze().transpose([1, 2, 0])
            img1 = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
            img2 = image_s[1].squeeze().transpose([1, 2, 0])
            img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
            msk1 = shot_mask_s[0].squeeze()
            msk2 = shot_mask_s[1].squeeze()
            qry = image_q.cpu().numpy().transpose([1, 2, 0])
            qry = (qry - np.min(qry)) / (np.max(qry) - np.min(qry))
            qry_msk1 = shot_mask_q[0]
            qry_msk2 = shot_mask_q[1]

            plt.figure()
            plt.subplot(3, 2, 1)
            plt.imshow(img1)
            plt.title("support img1")
            plt.axis("off")
            plt.subplot(3, 2, 2)
            plt.imshow(msk1)
            plt.title("support msk1 -- cls: " + str(cls[0]))
            plt.axis("off")
            plt.subplot(3, 2, 3)
            plt.imshow(img2)
            plt.title("support img2")
            plt.axis("off")
            plt.subplot(3, 2, 4)
            plt.imshow(msk2)
            plt.title("support msk2 -- cls: " + str(cls[1]))
            plt.axis("off")
            plt.subplot(3, 3, 7)
            plt.imshow(qry)
            plt.title("query img")
            plt.axis("off")
            plt.subplot(3, 3, 8)
            plt.imshow(qry_msk1)
            plt.title("query msk -- cls: " + str(cls[0]))
            plt.axis("off")
            plt.subplot(3, 3, 9)
            plt.imshow(qry_msk2)
            plt.title("query msk -- cls: " + str(cls[1]))
            plt.axis("off")
            path = "./2_way_sample_fold" + str(fold) + "_[" + str(i) + "].jpg"
            plt.savefig(path)
            print("demo image save to %s." % path)


    ###########################
    # run the functions below #
    ###########################

    run_VOCSBDSegmentation5i_from_scratch()
    # run_VOCSBDSegmentation5i_wrapper()
    # run_WrappedVOCSBDSegmentation5i_dataloader()
    # run_WrappedVOCSBDSegmentation5_dataloader()
    # run_WrappedVOCSBDSegmentation5i_network()
    # run_WrappedVOCSBDSegmentation5_network()

    # run_VOCSBDSegmentation5i_rebuild_pair()

    # run_VOCSBDSegmentation5i_2_way_images()
