from .voc_sbd import WrappedVOCSBDSegmentation5i
from .voc_sbd import WrappedVOCSBDSegmentation5
from .coco import WrappedCOCOStuff20i
from .coco import WrappedCOCOStuff20


def get_loader(name):
    """get_loader"""
    print("Using dataset: {}".format(name))
    return key2loader[name]


key2loader = {
    'WrappedVOCSBDSegmentation5i': WrappedVOCSBDSegmentation5i,
    'WrappedVOCSBDSegmentation5': WrappedVOCSBDSegmentation5,
    'WrappedCOCOStuff20i': WrappedCOCOStuff20i,
    'WrappedCOCOStuff20': WrappedCOCOStuff20,
}
