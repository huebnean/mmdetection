from .coco import CocoDataset
from .builder import DATASETS
import json
from pathlib import Path

@DATASETS.register_module()
class BDD100kDataset(CocoDataset):
    CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle', 'traffic light', 'traffic sign')

    PALETTE = () #TODO