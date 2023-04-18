from .coco import CocoDataset
from .builder import DATASETS
import json
from pathlib import Path

@DATASETS.register_module()
class A2D2Dataset(CocoDataset):
    CLASSES = ('car', 'pedestrian')

    PALETTE = () #TODO


