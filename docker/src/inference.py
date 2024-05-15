import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import single_gpu_test, inference_detector, show_result_pyplot
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import  build_dp
from pathlib import Path
import json


DEFAULT_CONFIG='configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
DEFAULT_CHECKPOINT='checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

def configure_cfg(cfg, dataset_folder):
    """ Configure the base config to add the dataset folders path

    Args:
        cfg (Config): Base Config file
        dataset_folder (str): Path to the dataset folder.
    Returns:
        _type_: _description_
    """

    # Remove the Loading annotation from the config pipeline, since inference does not have annotations
    updated_test_pipeline = []
    for item in cfg.data.test.pipeline:
        if item['type'] == "LoadAnnotations":
            continue
        updated_test_pipeline.append(item)


    # Modify dataset related settings
    test_data=dict(
            type='CocoDataset',
            img_prefix="",
            pipeline=updated_test_pipeline,
            ann_file=dataset_folder + "/results/annotations/converted_labels.json")
    cfg.data.test = test_data
    return cfg

def convert_and_save(dataset_folder, inference_results, classes):
    """Convert  the inference results to format need for the KPI scripts and save it

    Args:
        dataset_folder (str): Path to the dataset folder.
        inference_results (list(list(floats))): Inference results
        classes (list(str)): List of defined classes by the model used
    """
    images = [item.stem for item in list(Path(dataset_folder + "/images").glob('*.jpg'))]
    json_dict = {}
    for img, inference_result in zip(images, inference_results):
        results_list = []
        for ind, instances in enumerate(inference_result):
            instance_class = classes[ind]
            for instance in instances:
                results_list.append({"box": [float(val) for val in instance[:-1]], "class": instance_class,
                                     "confidence": float(instance[-1])})
        json_dict[img] = results_list

    with open(dataset_folder + '/results/result.json', 'w') as fp:
        json.dump(json_dict, fp)


def inference(dataset_folder, config=DEFAULT_CONFIG, checkpoint=DEFAULT_CHECKPOINT):
    """_summary_
    dataset_folder (str): Path to the dataset folder.
    config (str): The config you want to select.
    checkpoint (str): Checkpoint file path
    """

    # Set the device to be used for evaluation
    device='cpu'

    # Load the config
    config = mmcv.Config.fromfile(config)
    updated_config = configure_cfg(config, dataset_folder)

    # build the dataloader
    dataset = build_dataset(updated_config.data.test)
    data_loader = build_dataloader(dataset, samples_per_gpu=1,
                     workers_per_gpu=1,
                     num_gpus=1,
                     dist=False,)


    # Set pretrained to be None since we do not need pretrained model here
    updated_config.model.pretrained = None

    # Initialize the detector
    model = build_detector(updated_config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = updated_config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    model = build_dp(model, device)
    outputs = single_gpu_test(model, data_loader)
    convert_and_save(dataset_folder, outputs, checkpoint['meta']['CLASSES'])


