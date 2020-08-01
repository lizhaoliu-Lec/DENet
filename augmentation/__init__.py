from .augmentations import *

key2aug = {
    'gamma': AdjustGamma,
    'hue': AdjustHue,
    'brightness': AdjustBrightness,
    'saturation': AdjustSaturation,
    'contrast': AdjustContrast,
    'rcrop': RandomCrop,
    'hflip': RandomHorizontallyFlip,
    'vflip': RandomVerticallyFlip,
    'scale': Scale,
    'rscale': RandomizedScale,
    'rsize': RandomSized,
    'rsizecrop': RandomSizedCrop,
    'rotate': RandomRotate,
    'translate': RandomTranslate,
    'ccrop': CenterCrop,
    'sscale': SquareScale,
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        print("Using No Augmentations")
        return None

    augs = []
    for aug_key, aug_param in aug_dict.items():
        augs.append(key2aug[aug_key](aug_param))
        print("Using Augmentations: {} with params {}".format(aug_key, aug_param))
    return Compose(augs)
