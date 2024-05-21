import json
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

DEFAULT_CONFIG='configs/yolox/yolox_tiny_8x8_300e_coco.py'
DEFAULT_CHECKPOINT='checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

def configure_cfg(cfg, dataset_folder, classes):
    """ Configure the base config to add the dataset folders path

    Args:
        cfg (Config): Base Config file
        dataset_folder (str): Path to the dataset folder.
        classes (list): List of all classes in the bdd100k dataset
    Returns:
        Config: Updates config
    """

    # Configuring the model
    cfg.model.bbox_head.num_classes = len(classes)

    # Modify dataset related settings
    cfg.train_dataset.dataset.ann_file = dataset_folder + "train/results/annotations/converted_labels.json"
    cfg.train_dataset.dataset.img_prefix = ""
    cfg.train_dataset.dataset.classes = classes

    dataset_type = 'CocoDataset'
    data = dict(
        samples_per_gpu=1,
        workers_per_gpu=1,
        persistent_workers=False,
        train=cfg.train_dataset,
        val=dict(
            type =dataset_type,
            classes=classes,
            img_prefix="",
            pipeline=cfg.test_pipeline,
            ann_file = dataset_folder + "val/results/annotations/converted_labels.json"),
        test=dict(
            type =dataset_type,
            classes=classes,
            img_prefix="",
            pipeline=cfg.test_pipeline,
            ann_file=dataset_folder + "val/results/annotations/converted_labels.json"))
    cfg.data = data
    cfg.gpu_ids = ['0']
    cfg.seed = 1
    cfg.work_dir = dataset_folder + "/results/"
    return cfg

def read_classes_from_json(categories_json):
    """Read the classes from the json file

    Args:
        categories_json (str): Path to the json file containing class information

    Returns:
        list: List of all classes in the bdd100k dataset
    """
    classes = []
    categories_dict = json.load(open(categories_json))
    for item in categories_dict['categories']:
        classes.append(item['name'])
    return classes

def train(dataset_folder, classes_json, config=DEFAULT_CONFIG, checkpoint=DEFAULT_CHECKPOINT):
    """Train the model on given dataset

    Args:
        dataset_folder (str): Path to the dataset folder
        classes_json (str): Path to the json file containing class information in the bdd100k dataset
        config (str, optional): Path to the training config filr. Defaults to DEFAULT_CONFIG.
        checkpoint (str, optional): Path to the checkpoint from which training is to be resumed. Defaults to DEFAULT_CHECKPOINT.
    """
    # read classes from json
    classes = read_classes_from_json(classes_json)

    # Set the device to be used for evaluation
    device='cpu'

    # Load the config
    config = mmcv.Config.fromfile(config)
    updated_config = configure_cfg(config, dataset_folder, classes)

    updated_config.device = device
    # Build dataset
    datasets = [build_dataset(updated_config.data.train)]

    # Build the detector
    model = build_detector(updated_config.model)
    updated_config.resume_from = checkpoint

    # Add an attribute for visualization convenience
    model.CLASSES = classes

    # train the detector
    train_detector(model, datasets, updated_config, distributed=False, validate=True)

train("downloads/", "downloads/bdd_10k_categories.json")


