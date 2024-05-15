import kpi.acl as precision_loss
from dataset import ImageDetectionDataset
from utils.utils import *



class KpiCalculator():

    instance = None

    def __new__(cls, *args, **kwargs):

        instance = KpiCalculator.instance

        if instance is None:

            instance = object.__new__(cls)
        return instance

    @classmethod
    def get_instance(cls) :

        if cls.instance is not None:

            return cls.instance

        else:
            instance = cls()
            return instance



    def __init__(self):

        if  KpiCalculator.instance is None:
            KpiCalculator.instance = self


            self.mAP = 0
            self.scen_cov = 0

    def precision_loss_calculation(self, reference_dataset: ImageDetectionDataset, reference_dataset_infer: ImageDetectionDataset,
                                   manipulated_datasets_infer: list, filter: dict = {},
                                   iou_threshold: float = 0.5, ignore_no_detection: bool = False,
                                   ignore_conf_gain: bool = True, object_source: list = [], min_bb: list = [0, 0], use_gt: bool = False, result_path: str = "",
                                   save_csv: bool = False, model_name: str = ""):

        precision_loss_results = precision_loss.compute_metric(reference_dataset, reference_dataset_infer, manipulated_datasets_infer,
                                                                   filter, iou_threshold, ignore_conf_gain, ignore_no_detection, object_source, min_bb, use_gt, result_path)
        if save_csv:
            precision_loss.save_results_as_csv(precision_loss_results, reference_dataset, model_name, ["car"], result_path)

        return precision_loss_results
        
if __name__ == '__main__':
    pass