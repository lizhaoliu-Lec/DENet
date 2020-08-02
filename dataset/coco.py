import copy
import os
import pickle
import zipfile
from collections import OrderedDict

import numpy as np
import scipy.misc as misc
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from augmentation import SquareScale, RandomHorizontallyFlip, Compose
from dataset.transforms import ToTensor, Normalize
from utils import replace_array_ele_as_dict, sort_dict_by

# freeze seed for reproducity
np.random.seed(1234)

is_dir = os.path.isdir
exists = os.path.exists

DATASET_SIZE_DIR = {
    '10k': {
        'url': 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip',
        'filename': 'cocostuff-10k-v1.1.zip',
    }
}

COCO14_DATASET_LINK = {
    'train': {
        'url': 'http://images.cocodataset.org/zips/train2014.zip',
        'filename': 'train2014.zip'
    },
    'val': {
        'url': 'http://images.cocodataset.org/zips/val2014.zip',
        'filename': 'val2014.zip'
    },
    'annotations': {
        'url': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'filename': 'annotations_trainval2014.zip'
    }
}


class COCO14(VisionDataset):
    """
    COCO 2014 Dataset <https://arxiv.org/pdf/1405.0312.pdf> for instance segmentation.

    Notes:
        This is only a subset of original COCO 2014, specifcally, we only support
        ``train`` split and ``val`` split for few shot segmentation task. In total,
        the ``train`` split has 82783 images and the ``val`` split has 40504 images.

    This dataset contains photos of 81 objects types that would be easily recognizable
    by a 4 year old. With a total of 2.5 million labeled instances in 328k images, the
    creation of our dataset drew upon extensive crowd worker involvement via novel user
    interfaces for category detection, instance spotting and instance segmentation.

    Args:
        root: root directory of the VOC Dataset.
        image_set: select the image_set to use, `train` or `val`.
        download: if true, download the dataset from the internet and put it in root.
            directory. If the dataset is already downloaded, it should not be downloaded.
        transform: a function/transform that takes in an PIL image and returns a transformed
            version. e.g, `transforms.RandomCrop`.
        target_transform: a function/transform that takes in the target and transforms it.
        transforms: a function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    ID2CLASS_NAME = {
        0: 'unlabeled',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        12: 'street sign',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        26: 'hat',
        27: 'backpack',
        28: 'umbrella',
        29: 'shoe',
        30: 'eye glasses',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        45: 'plate',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        66: 'mirror',
        67: 'dining table',
        68: 'window',
        69: 'desk',
        70: 'toilet',
        71: 'door',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        83: 'blender',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush',
        91: 'hair brush',
    }

    def __init__(self,
                 root,
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(COCO14, self).__init__(root, transforms, transform, target_transform)
        self.dataset_url = COCO14_DATASET_LINK[image_set]['url']
        self.dataset_filename = COCO14_DATASET_LINK[image_set]['filename']
        self.annots_url = COCO14_DATASET_LINK['annotations']['url']
        self.annots_filename = COCO14_DATASET_LINK['annotations']['filename']
        self.image_set = image_set
        self.images_dir = os.path.join(self.root, 'images')
        self.images_base_dir = os.path.join(self.images_dir, self.image_set + '2014')
        self.masks_base_dir = os.path.join(self.root, 'annotations')
        self.train_masks_file = os.path.join(self.masks_base_dir, 'instances_train2014.json')
        self.val_masks_file = os.path.join(self.masks_base_dir, 'instances_val2014.json')

        if download:
            print("\nBegin downloading COCO14 %s dataset..." % self.image_set)
            download_extract(self.dataset_url, self.images_dir, self.dataset_filename, md5=None)
            if exists(self.train_masks_file) and exists(self.val_masks_file):
                print("Annotations files already exist.")
            else:
                print("Begin downloading COCO14 train/val annotations...")
                download_extract(self.annots_url, self.root, self.annots_filename, md5=None)
            print("Download completed.")

        if not (is_dir(self.images_base_dir)
                and is_dir(self.masks_base_dir)
                and exists(self.train_masks_file)
                and exists(self.val_masks_file)):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        idxes = os.listdir(self.images_base_dir)
        self.images = [os.path.join(self.images_base_dir, idx) for idx in idxes]
        self.masks = list()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class COCO1480(COCO14):
    """
    COCO 2014-80 Segmentation Dataset. Select the 80 classes + unlabeled as mentioned in
    Feature Weighting and Boosting for Few-Shot Segmentation <https://arxiv.org/pdf/1909.13140.pdf>
    Resulting in that the total number of classes is 81 and the labels are range from 0 to 80.

    Notes:
        In total, the ``train`` split has 82783 images and the ``val`` split has 40504 images.

    Args:
        replace_with: replace the classes's mask other than the 81 classes with the number
            specified by this argument, default is 0.
        rebuild_mask: whether to rebuild the mask (only consist of 81 classes).
    """
    EIGHTY_CLASSES = {
        0: 'unlabeled',
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella shoe',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink ',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush',
    }

    def __init__(self,
                 replace_with=0,
                 rebuild_mask=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_with = replace_with
        self.rebuild_mask = rebuild_mask
        self.masks80_base_dir = os.path.join(self.root, "COCO14_80Mask", self.image_set)
        self.sorted_id_class_pairs = sort_dict_by(self.EIGHTY_CLASSES)
        self.oldId2newId = {old: new for new, (old, _) in enumerate(self.sorted_id_class_pairs)}
        self.newId2oldId = {new: old for old, new in self.oldId2newId.items()}

        if not is_dir(self.masks80_base_dir) or rebuild_mask:
            self._convert_instance2semantic()

        mask_paths = os.listdir(self.masks80_base_dir)
        self.idxes = [os.path.basename(p).split('.')[0] for p in mask_paths]
        idxes = self.idxes
        self.images = [os.path.join(self.images_base_dir, '%s.jpg' % idx) for idx in idxes]
        self.masks = [os.path.join(self.masks80_base_dir, '%s.png' % idx) for idx in idxes]
        assert (len(self.images) == len(self.masks)), \
            'images len: %d, masks len: %d' % (len(self.images), len(self.masks))

    def _convert_instance2semantic(self):
        # Define paths
        ann_path = os.path.join(self.masks_base_dir, "instances_" + self.image_set + "2014.json")
        png_folder = self.masks80_base_dir

        # Create output folder
        if not os.path.exists(png_folder):
            os.makedirs(png_folder)

        # Initialize COCO ground-truth API
        coco = COCO(ann_path)
        img_ids = coco.getImgIds()

        # Convert each image to a png
        print("\nBegin converting instance masks to semantic masks...")
        img_count = len(img_ids)
        img_loader = tqdm(range(0, img_count), position=0, ncols=125)
        for i in img_loader:
            img_id = img_ids[i]
            img_name = coco.loadImgs(ids=img_id)[0]['file_name'].replace('.jpg', '')
            segmentation_path = os.path.join(png_folder, img_name + '.png')
            self._convert_to_png(coco, img_id, segmentation_path, convert_labels=True)
            img_loader.set_description('image %d of %d: %s' % (i + 1, img_count, img_name))
        print("Converting completed.")

    def _convert_to_png(self, coco, img_id, png_path, include_crowd=False, convert_labels=True):
        # Create label map
        label_map = self._to_seg_map(coco, img_id, False, include_crowd=include_crowd)
        label_map = np.array(label_map, dtype=np.int8)

        # Ensure each png labels range 0~81
        if convert_labels:
            label_map = replace_array_ele_as_dict(label_map, self.oldId2newId, self.replace_with)

        # Convert array to gray image and write to png file
        png = Image.fromarray(label_map).convert('L')
        png.save(png_path, format='PNG')

    @staticmethod
    def _to_seg_map(coco, img_id, check_unique_pixel_label=True, include_crowd=False):
        # Init
        cur_img = coco.imgs[img_id]
        image_size = (cur_img['height'], cur_img['width'])
        label_map = np.zeros(image_size)

        # Get annotations of the current image (may be empty)
        imgAnnots = [a for a in coco.anns.values() if a['image_id'] == img_id]
        if include_crowd:
            annIds = coco.getAnnIds(imgIds=img_id)
        else:
            annIds = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        imgAnnots = coco.loadAnns(annIds)

        # Combine all annotations of this image in labelMap
        for a in range(0, len(imgAnnots)):
            label_mask = coco.annToMask(imgAnnots[a]) == 1
            new_label = imgAnnots[a]['category_id']

            if check_unique_pixel_label and (label_map[label_mask] != 0).any():
                raise Exception('Error: Some pixels have more than one label (image %d)!' % img_id)

            label_map[label_mask] = new_label

        return label_map

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class COCO1420(COCO1480):
    """
    COCO 2014-20 Segmentation Dataset. Select the 60 classes + unlabeled during training,
    and all the 81 classes in test. For more detail how to hold out the 20 classes for each fold,
    please refer to Feature Weighting and Boosting for Few-Shot Segmentation <https://arxiv.org/pdf/1909.13140.pdf>.
    We construct it based on COCO 2014-80 dataset.

    Notes:
        In total, the ``train`` split has 82783 images and the ``val`` split has 40504 images.

    Args:
        image_set:
            train: 61 classes from COCO 2014-80 train set
            test: 81 classes from COCO 2014-80 test set
        fold: whether to construct train, test as fold style. If None, return to normal setting.
        rebuild_fold_mask: whether to rebuild fold mask, only useful when `fold` is not None.
        ignore_idx: integer to fill the ignore places.
    """

    def __init__(self,
                 fold=None,
                 image_set='train',
                 rebuild_fold_mask=False,
                 ignore_idx=255,
                 *args, **kwargs):
        # COCO14 20 dataset only supports train split and val split
        if image_set == 'test':
            image_set = 'val'
        super().__init__(image_set=image_set, *args, **kwargs)
        self.image_set = image_set
        self.fold = fold
        self.rebuild_fold_mask = rebuild_fold_mask
        self.ignore_idx = ignore_idx
        self.fold_masks_base_dir = os.path.join(self.root, 'fold%s' % self.fold, self.image_set)
        assert len(self.images) == len(self.masks), \
            'images len: %d, masks len: %d' % (len(self.images), len(self.masks))

        if self.fold is not None:
            self._fold()
            assert len(self.images) == len(self.masks), \
                'images len: %d, masks len: %d' % (len(self.images), len(self.masks))

    def _fold(self):
        fold_mask_root = self.fold_masks_base_dir
        if not os.path.exists(fold_mask_root) or self.rebuild_mask:
            os.makedirs(fold_mask_root, exist_ok=True)
            fold_images = list()
            fold_masks = list()
            ignore_classes = self.get_classes_to_ignore()
            ignore_classes2ignore_idx = {ic: self.ignore_idx for ic in ignore_classes}
            if self.image_set == 'train':
                print('Ignoring classes for fold %d in split %s: ' % (self.fold, self.image_set), ignore_classes)
            else:
                print('Preserving all classes for fold %d in split %s' % (self.fold, self.image_set))
            print('Preparing data for fold %d in split %s' % (self.fold, self.image_set))
            for index in tqdm(range(len(self)), position=0, leave=True):
                original_mask_path = self.masks[index]
                fold_mask_path = os.path.join(fold_mask_root,
                                              os.path.basename(original_mask_path))
                target = Image.open(original_mask_path)
                target = np.array(target, dtype=np.int)
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
            mask_path = [os.path.join(fold_mask_root, i) for i in os.listdir(fold_mask_root)]
            self.masks = mask_path
            new_images = []
            for m_path in mask_path:
                image_path = os.path.join(self.images_base_dir, os.path.basename(m_path).split('.')[0] + '.jpg')
                assert os.path.exists(m_path), 'mask %s not exist' % m_path
                assert os.path.exists(image_path), 'image %s not exist' % image_path
                new_images.append(image_path)
            self.images = new_images

    def get_classes_to_ignore(self):
        ignore_when_train = [_ for _ in range(self.fold + 1, 81, 4)]
        return ignore_when_train

    @staticmethod
    def get_mask_fraction_has_at_least_threshold(mask, threshold, ignore_idx):
        m = mask.flatten()
        from collections import Counter
        id2cnt = Counter(m)
        total = sum(id2cnt.values())
        id2fra = {i: c / total for i, c in id2cnt.items()}
        id2fra = {i: f for i, f in id2fra.items() if f >= threshold}
        # always preserve ignore and background
        return list(id2fra.keys()) + [ignore_idx, 0]


class COCOStuff20i(COCO1420):
    """
    COCOStuff20i Segmentation Dataset. Comparing with COCOStuff20, this dataset also return
    the binary mask according to the query class in a pair style.

    Args:
        train_size: due to the fact that we are randomly sample the pair to construct the final
            dataset, we can sample up to O(n^2) image to train the model, which makes it into
            64~million pairs and may not be feasible if you don't need that, so just specify the
            train_size. We sample the classes in a balanced way, the final classes may has a
            smaller number of images & masks.
        test_size: analogy to train_size, but specify for test.
        rebuild_pair: whether to rebuild query support pair.
    """

    def __init__(self,
                 train_size=60000,
                 test_size=1000,
                 rebuild_pair=False,
                 rebuild_whole_pair=False,
                 shot=1,
                 way=1,
                 *args, **kwargs):
        self.support_query = None
        self.train_size = train_size
        self.test_size = test_size
        self.threshold = 0.01
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
        item_class_dir = os.path.join(self.root, 'item_with_class', self.image_set, str(self.fold))
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
        ignore_classes = self.get_classes_to_ignore()

        item_class_dir = os.path.join(self.root, 'item_with_class', self.image_set, str(self.fold),
                                      str(self.threshold))
        if not os.path.exists(item_class_dir):
            os.makedirs(item_class_dir)
        item_class_pair = 'item_class.pair'
        if self.image_set == 'val' and self.way > 1:
            item_class_pair = 'unlimited_item_class.pair'
        item_class_path = os.path.join(item_class_dir, item_class_pair)
        if not os.path.exists(item_class_path) or self.rebuild_pair:
            item_class = list()
            print('Caching item class pair to %s' % item_class_path)
            for index in tqdm(range(super().__len__()), position=0, leave=True):
                image_path, mask_path = self.images[index], self.masks[index]
                _, mask = super().__getitem__(index)
                mask = np.array(mask, dtype=np.int)
                uniques = self.get_mask_fraction_has_at_least_threshold(mask, self.threshold, self.ignore_idx)
                if self.image_set == 'train':
                    # depart the root dir considering the portability
                    item_class.extend([(file_name(image_path), file_name(mask_path), c) for c in uniques if
                                       c != self.ignore_idx and c != 0])
                elif self.image_set == 'val' and self.way == 1:
                    # we only select the images that have base classes and only 1 novel class
                    old_length = len(set(ignore_classes))
                    if len(set(ignore_classes) - set(uniques)) != old_length - 1:
                        continue
                    # depart the root dir considering the portability
                    item_class.extend([(file_name(image_path), file_name(mask_path), c) for c in uniques if
                                       c in ignore_classes])
                else:
                    # i.e., image_set == 'val' and way > 1
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
        size = self.train_size if self.image_set == 'train' else self.test_size
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
            if len(items) ** 2 < num and self.image_set != 'val':
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
        join = os.path.join
        return join(self.images_base_dir, img_path), join(self.fold_masks_base_dir, mask_path)

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


class WrappedCOCOStuff20(COCOStuff20i):
    """
    WrappedCOCOStuff20 dataset for normal segmentation task.

    Args:
        split: same as image_set, but with different name.
        img_size: expected img_size for both image and mask.
    """

    def __init__(self, root, fold, split, img_size, transforms=None, *args, **kwargs):
        if transforms is None:
            transforms = Compose([
                SquareScale(size=img_size),
                RandomHorizontallyFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])
        super().__init__(root=root,
                         fold=fold,
                         image_set=split,
                         transforms=transforms,
                         *args, **kwargs)
        self.support_query = None
        self.query = self.construct_query()

    def construct_query(self):
        unique_cls = sorted(list(set([item[-1] for item in self.item_class])))
        size = self.train_size if self.image_set == 'train' else self.test_size
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


class WrappedCOCOStuff20i(COCOStuff20i):
    """
    WrappedCOCOStuff20i dataset for easy out of the box.

    Args:
        split: same as image_set, but with different name.
        img_size: expected img_size for both image and mask.
    """

    def __init__(self, root, fold, split, img_size, transforms=None, *args, **kwargs):
        if transforms is None:
            transforms = Compose([
                SquareScale(size=img_size),
                RandomHorizontallyFlip(p=0.5),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])
        super().__init__(root=root,
                         fold=fold,
                         image_set=split,
                         transforms=transforms,
                         *args, **kwargs)

    @staticmethod
    def to_cuda(*args):
        return [a.cuda() for a in args]


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    filepath = os.path.join(root, filename)
    if zipfile.is_zipfile(filepath):
        fz = zipfile.ZipFile(filepath, 'r')
        for file in fz.namelist():
            fz.extract(file, root)
    else:
        print('Can not extract %s' % filepath)


if __name__ == '__main__':
    #################################################################
    # The following functions are just some basic test functions
    # to see if the codes work. But to avoid the annoying
    # Pycharm auto test triggered, we use the `run` instead of `test`.
    ##################################################################

    # specify root path for the data path
    ROOT = "/path/to/data"
    root_path = os.path.join(ROOT, 'COCO')


    def run_COCOStuff20i_from_scratch():
        fold = 0
        coco = COCOStuff20i(root=root_path,
                            image_set='train',
                            fold=fold,
                            download=True)
        print('split: train, fold: %d, len: ' % fold, len(coco))

        coco = COCOStuff20i(root=root_path,
                            image_set='test',
                            fold=fold,
                            download=False)  # already download above
        print('split: test, fold: %d, len: ' % fold, len(coco))


    def run_WrappedCOCOStuff20i():
        dataset = WrappedCOCOStuff20i(root=root_path,
                                      fold=0,
                                      split='train',
                                      img_size=224,
                                      shot=1,
                                      way=1)
        idxes = [_ for _ in range(0, len(dataset))]
        import random
        random.shuffle(idxes)

        for i in idxes[:10]:
            image_s, shot_mask_s, image_q, shot_mask_q, cls, mask_s, mask_q = dataset[i]
            print(image_s.shape, shot_mask_s.shape, image_q.shape, shot_mask_q.shape,
                  cls, mask_s.shape, mask_q.shape)


    def run_WrappedCOCOStuff20i_dataloader():
        from torch.utils import data
        fold = 0
        split = 'train'
        dataset = WrappedCOCOStuff20i(root=root_path,
                                      fold=fold,
                                      split=split,
                                      img_size=224)

        print("len: ", len(dataset))

        train_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
        for d in train_loader:
            image_s, shot_mask_s, image_q, shot_mask_q, cls, mask_s, mask_q = d
            print(image_s.shape, shot_mask_s.shape, image_q.shape, shot_mask_q.shape,
                  cls, mask_s.shape, mask_q.shape)


    def run_WrappedCOCOStuff20_dataloader():
        from torch.utils import data
        dataset = WrappedCOCOStuff20(root=root_path,
                                     fold=0,
                                     split='train',
                                     img_size=224)
        train_loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

        for d in train_loader:
            imgs, masks = d
            print(imgs.shape, masks.shape)


    def run_COCOStuff20i_2_way_images():
        fold = 0
        dataset = WrappedCOCOStuff20i(root=root_path,
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


    def run_WrappedCOCOStuff20i_network():
        from torch.utils import data
        from model.head.pgn import PGN
        import torch.nn.functional as F
        from loss import cross_entropy2d
        from optimizer import get_optimizer

        batch_size = 4
        epoch = 1

        train_set = WrappedCOCOStuff20i(root=root_path,
                                        fold=0,
                                        # remember to run both train and val set
                                        split='train',
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


    def run_WrappedCOCOStuff20_network():
        from torch.utils import data
        from model.head.amp import AMP
        import torch.nn.functional as F
        from loss import cross_entropy2d
        from optimizer import get_optimizer

        batch_size = 4
        epoch = 1

        train_set = WrappedCOCOStuff20(root=root_path,
                                       fold=0,
                                       split='train',
                                       rebuild_mask=False,
                                       img_size=224)
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

        model = AMP(maximum_num_classes=81)

        optim = get_optimizer("rmsprop")(model.parameters(), lr=0.000001,
                                         weight_decay=0.0005)

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


    ###########################
    # run the functions below #
    ###########################

    run_COCOStuff20i_from_scratch()
    # run_WrappedCOCOStuff20i()
    # run_WrappedCOCOStuff20i_dataloader()
    # run_WrappedCOCOStuff20_dataloader()
    # run_COCOStuff20i_2_way_images()
    # run_WrappedCOCOStuff20i_network()
    # run_WrappedCOCOStuff20_network()
