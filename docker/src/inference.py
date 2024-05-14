import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import glob

DEFAULT_CONFIG='configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py'
DEFAULT_CHECKPOINT='checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

def inference(img, config=DEFAULT_CONFIG, checkpoint=DEFAULT_CHECKPOINT):
    """_summary_
    img (str or list[str]): Image files path on which inference is to be performed.
    config (str): The config you want to select.
    checkpoint (str): Checkpoint file path
    """

    # Set the device to be used for evaluation
    device='cpu'

    # Load the config
    config = mmcv.Config.fromfile(config)

    # Remove the Loading annotation from the config pipeline, since inference does not have annotations
    updated_test_pipeline = []
    for item in config.data.test.pipeline:
        if item['type'] == "LoadAnnotations":
            continue
        updated_test_pipeline.append(item)

    config.data.test.pipeline = updated_test_pipeline

    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    # Use the detector to do inference
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result, score_thr=0.3)


