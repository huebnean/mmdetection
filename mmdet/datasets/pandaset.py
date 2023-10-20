
from .builder import DATASETS
from .coco import CocoDataset
from .custom import CustomDataset

@DATASETS.register_module()
class PandasetDataset(CustomDataset):

    CLASSES = []

    PALETTE = []