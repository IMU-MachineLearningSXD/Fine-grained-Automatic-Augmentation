# from ._360cc import _360CC
from .bigdata_own import _OWN


def get_dataset(config):

    # if config.DATASET.DATASET == "360CC":
    #     return _360CC
    if config.DATASET.DATASET == "OWN":
        return _OWN
    else:
        raise NotImplemented()