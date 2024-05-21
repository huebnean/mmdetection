import json
import os
from pathlib import Path
from datetime import datetime
import math as math
from PIL import Image
import numpy as np
from builtins import list

base = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")

class Dataset:
    """
    this class is the base abstract  class for Dataset
    """

    def __init__(self, DatasetDict: dict, GroundTruth: dict, Name: str = "DataSet"):
        """
        constructor for the abstract class of Dataset


        Input:
            DatasetDict : Dictionary of information with keys
            GroundTruth : Dictionary of corresponding GroundTruth information with keys
            Name : Name of the dataset

        Output:
            object of the class Dataset

        """

        self.DatasetDict = DatasetDict
        self.GroundTruth = GroundTruth
        self.Name = Name



    def get(self, key: str):
        """
        get the data from  DatasetDict with key


        Input:
            key : string with the key that corresponds the the the info


        Output:
            the info saved in  DatasetDict

        """

        return self.DatasetDict.get(key)

    def keys(self):
        """
        return a list of Available keys


        Input:
            None


        Output:
            list of Available keys

        """

        return list(self.DatasetDict.keys())

    def get_Ground_truth(self, key: str):
        """
        get the data from  GroundTruth with key

        Input:
            key : string with the key that corresponds the the the info

        Output:
            the info saved in  GroundTruth

        """

        return self.GroundTruth.get(key)

    def add(self, key: str, Data, Info):
        """
        add  info  to  GroundTruth and DatasetDict with key

        Input:
            key : string with the key that corresponds the the the info
            Data :  the info to be saved in  DatasetDict
            Info :  the info to be saved in  GroundTruth

        Output:
            None

        """

        self.DatasetDict[key] = Data
        self.GroundTruth[key] = Info



class ImageDetectionDataset(Dataset):
    """
    This class inherits from Dataset and contains the necessary behaviors for image recognition

    """

    def __init__(self, DatasetDict: dict = None, GroundTruth: dict = None, Name: str = "dataset",
                 semantic_segmentationDict: dict = dict(), DatasetInfo: dict = dict()):
        """
        constructor for the ImageDetectionDataset class


        Input:
            DatasetDict : Dictionary of image path with keys
            GroundTruth : Dictionary of corresponding GroundTruth Bounding boxes information with keys
            semantic_segmentationDict : Dictionary of corresponding semantic_segmentation with keys
            Name : Name of the dataset

        Output:
            object of the class Dataset
        """

        Dataset.__init__(self, DatasetDict, GroundTruth, Name)

        self.semantic_segmentationDict = semantic_segmentationDict
        self.DatasetInfo = DatasetInfo

    instance = None

    def __new__(cls, *args, **kwargs):
        """singleton pattern"""

        instance = ImageDetectionDataset.instance

        if instance is None:
            instance = object.__new__(cls)
        return instance

    @classmethod
    def get_instance(cls):
        """singleton pattern"""

        if cls.instance is not None:

            return cls.instance

        else:
            instance = cls()
            return instance

    def get_dataset_info(self):
        """
        Return Dataset information as dictionary back


        Input:
            None

        Output:
            dictionary Contining   the dataset  name and number of its key
        """

        return {"DataSetName": self.Name, "N_keys": len(self.keys())}


    def get(self, key: str) -> Image:
        """
        get Image obj  from  DatasetDict with key


        Input:
            key : string with the key that corresponds the the the info


        Output:
            the Image wicht its path is saved in  DatasetDict

        """
        image_path = Dataset.get(self, key)

        image = None
        if image_path is not None and os.path.isfile(image_path):
            try:
                Image.open(image_path)
            except:
                print("Image with the following path is not available: ", image_path)
                return None

            with Image.open(image_path) as im:
                image = im.copy()
                image = image.convert('RGB')

                cam_name = 'front_center'
                image = np.array(image)

                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
                image = Image.fromarray(image)

        return image

    def add(self, key: str, Data, Info, segmentation=None):
        """
        add  info  to  GroundTruth and DatasetDict with key

        Input:
            key : string with the key that corresponds the the the info
            Data :   the path of the  image, to be saved in  DatasetDict
            Info :  the info to be saved in  GroundTruth
            segmentation : the path of the semantic segmentation image

        Output:
            None

        """

        Dataset.add(self, key, Data, Info)

        self.semantic_segmentationDict[str(key)] = segmentation


    def get_coco_dataset(coco_path):
        """
        import Dataset in Coco format

        Input:
            path : Path of the dataset

        Output:
            object of ImageDetectionDataset

        """

        dataset_dict = dict()
        ground_truth = dict()
        dataset_info = dict()

        image_id_to_image_name_dict = dict()
        category_id_to_category_name_dict = dict()

        directory, filename = os.path.split(coco_path)

        if len(filename.split(".")) > 0:

            dataset_name = filename.split(".")[0]

        else:

            dataset_name = filename

        # load info from coco annotation file TODO: search for anno file
        if(os.path.isfile(coco_path)):
            with open(coco_path) as f:
                coco_data = json.load(f)
        else:
            with open(os.path.join(coco_path, "coco.json")) as f:

                oco_data = json.load(f)

        # load categories
        for category in coco_data['categories']:
            category_id_to_category_name_dict[category['id']] = category['name']

        # store image names and paths in DatasetDict
        for image_data in coco_data['images']:
            image_path = image_data['file_name']
            image_name = Path(image_path).name.split('.')[0]

            dataset_dict[image_name] = image_path

            image_id_to_image_name_dict[image_data['id']] = image_name

        # store ground truth (cvt bbox and names)
        for coco_annotation in coco_data['annotations']:
            if coco_annotation['image_id'] not in image_id_to_image_name_dict:
                print('Image with the following id does not exist: ', coco_annotation['image_id'])
                continue

            if coco_annotation['category_id'] not in category_id_to_category_name_dict:
                print('Class with the following id does not exist: ', coco_annotation['category_id'])
                continue

            image_name = image_id_to_image_name_dict[coco_annotation['image_id']]
            if image_name not in ground_truth:
                ground_truth[image_name] = []

            annotation = dict()
            annotation['class'] = category_id_to_category_name_dict[coco_annotation['category_id']]
            annotation['box'] = [coco_annotation['bbox'][0], coco_annotation['bbox'][1],
                                 coco_annotation['bbox'][0] + coco_annotation['bbox'][2],
                                 coco_annotation['bbox'][1] + coco_annotation['bbox'][
                                     3]]  # x0, y0, w, h -> x0, y0, x1, y1

            # add other labels
            for label_key, label_value in coco_annotation.items():
                if label_key in ['id', 'image_id', 'segmentation', 'category_id', 'bbox']:
                    continue

                annotation[label_key] = label_value

            ground_truth[image_name].append(annotation)

        # generate dataset info dict
        dataset_info['original_dataset'] = dict()
        dataset_info['original_dataset']['dataset_name'] = dataset_name
        dataset_info['original_dataset']['size'] = len(dataset_dict)
        dataset_info['original_dataset']['is_synthetic'] = None
        dataset_info['original_dataset']['creation_duration'] = 0
        dataset_info['original_dataset']['creation_date'] = None

        # TODO: store semantic segmentation
        semantic_segmentation = None

        return ImageDetectionDataset(dataset_dict, ground_truth, dataset_name, semantic_segmentation, dataset_info)


    @classmethod
    def get_datasets_names(cls):
        """
        class method that returns a list of available Datasets Names

        Input:
            None

        Output:
            a list of available Datasets Names

        """

        mybase = os.path.join(base, "datasets")

        DatasetsNames = ["a2d2_dataset", "carla_dataset", "synthetic_dataset"]

        all_subdirs = [d for d in os.listdir(mybase) if os.path.isdir(os.path.join(mybase, d))]
        all_subdirs_inference_results = [d for d in os.listdir(os.path.join(mybase, "inference_results")) if
                                         os.path.isdir(os.path.join(mybase, "inference_results", d))]
        all_subdirs_manipulated_datasets = [d for d in os.listdir(os.path.join(mybase, "manipulated_datasets")) if
                                            os.path.isdir(os.path.join(mybase, "manipulated_datasets", d))]

        for dir_name in all_subdirs:

            # if dir_name not in DatasetsNames:

            path = os.path.join(mybase, dir_name)

            if os.path.exists(os.path.join(path, "./DatasetDict.json")) and os.path.exists(
                    os.path.join(path, "./GroundTruth.json")):
                DatasetsNames.append(dir_name)

        for dir_name in all_subdirs_manipulated_datasets:

            # if dir_name not in DatasetsNames:

            path = os.path.join(mybase, "manipulated_datasets", dir_name)

            if os.path.exists(os.path.join(path, "./DatasetDict.json")) and os.path.exists(
                    os.path.join(path, "./GroundTruth.json")):
                DatasetsNames.append(dir_name)

        for dir_name in all_subdirs_inference_results:

            # if dir_name not in DatasetsNames:

            path = os.path.join(mybase, "inference_results", dir_name)

            if os.path.exists(os.path.join(path, "./DatasetDict.json")) and os.path.exists(
                    os.path.join(path, "./GroundTruth.json")):
                DatasetsNames.append(dir_name)

        return list(set(DatasetsNames))


    @classmethod
    def get_datasets_by_name(cls, DatasetsName: str):
        """
        class method that returns a Image Detection Dataset object by its Name

        Input:
            DatasetsName : Dataset Name as string

        Output:
            object of the ImageDetectionDataset

        """

        if DatasetsName == "a2d2_dataset":

            return ImageDetectionDataset.get_a2d2_dataset()

        elif DatasetsName == "carla_dataset":

            return ImageDetectionDataset.get_carla_dataset()

        elif DatasetsName == "synthetic_dataset":

            return ImageDetectionDataset.get_synthetic_dataset()

        else:
            path = os.path.join(base, "datasets", DatasetsName)

            if os.path.isdir(path):
                dataset = ImageDetectionDataset.import_from_json(path)
                dataset.Name = DatasetsName

                return dataset

            path = os.path.join(base, "datasets", "inference_results", DatasetsName)

            if os.path.isdir(path):
                dataset = ImageDetectionDataset.import_from_json(path)
                dataset.Name = DatasetsName

                return dataset

            path = os.path.join(base, "datasets", "manipulated_datasets", DatasetsName)

            if os.path.isdir(path):
                dataset = ImageDetectionDataset.import_from_json(path)
                dataset.Name = DatasetsName

                return dataset

            path = os.path.join(base, "datasets", "coco_datasets", DatasetsName)

            if os.path.isdir(path):
                dataset = ImageDetectionDataset.get_coco_dataset(path)
                dataset.Name = DatasetsName

                return dataset

    @classmethod
    def delete_dataset_by_name(cls, DatasetsName: str):
        """
        class method that removes a Image Detection by its Name

        Input:
            DatasetsName : Dataset Name as string

        Output:
            None

        """

        path = os.path.join(base, "datasets", DatasetsName)
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True
        path = os.path.join(base, "datasets", "inference_results", DatasetsName)
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True

        path = os.path.join(base, "datasets", "manipulated_datasets", DatasetsName)
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True

        return False


    @classmethod
    def import_from_json(cls, path):
        """
        import Dataset info from path as json

        Input:
            path : Path where the results are imported

        Output:
            object of ImageDetectionDataset

        """
        try:

            pathsplit = path.split(os.sep)

            if len(pathsplit) > 0:

                Name = str(pathsplit[-1])

            else:

                Name = "ImageDetectionDataset"

            json_DatasetDict = os.path.join(path, "./DatasetDict.json")

            with open(json_DatasetDict, 'r') as fp:
                DatasetDict = json.load(fp)

            json_GroundTruth = os.path.join(path, "./GroundTruth.json")

            with open(json_GroundTruth, 'r') as fp:
                GroundTruth = json.load(fp)

            json_DatasetInfo = os.path.join(path, "./DatasetInfo.json")

            with open(json_DatasetInfo, 'r') as fp:
                DatasetInfo = json.load(fp)

            semantic_segmentation_DatasetDict = os.path.join(path, "./semantic_segmentation.json")

            if os.path.isfile(semantic_segmentation_DatasetDict):
                with open(json_DatasetDict, 'r') as fp:
                    semantic_segmentation = json.load(fp)
            else:

                semantic_segmentation = dict()

            return cls(DatasetDict, GroundTruth, Name, semantic_segmentation, DatasetInfo)
        except:

            return None

    @classmethod
    def filter_dataset_with_minimum_bounding_box(cls, DatasetObject, minimum_bounding_box_width: float,
                                                 minimum_bounding_box_hight: float):
        """
        class method that filter a Image Detection by the keys of another sub Dataset

        Input:
            DatasetObject : object of  ImageDetectionDataset --> the bigger dataset
            minimum_bounding_box_width : minimum bounding box width
            minimum_bounding_box_hight : minimum bounding box hight


        Output:
            resultsnew : object of  ImageDetectionDataset --> the sub dataset with bb > minimum_bounding_box

        """

        # print("filter_dataset__with_minimum_bounding_box")

        # DatasetObject = DatasetObject
        start_creation_time = datetime.now().timestamp()
        count = 0
        for key in DatasetObject.keys():
            count = count + 1
            GroundTruthList = DatasetObject.GroundTruth.get(key, [])

            # print(len(GroundTruthList))
            GroundTruthNewList = []

            for ele in GroundTruthList:
                [y1, x1 ,y2, x2 ] = ele.get('box')


                width = math.fabs(x2 - x1)
                hight = math.fabs(y2 - y1)

                if width >= minimum_bounding_box_width and hight >= minimum_bounding_box_hight:
                    GroundTruthNewList.append(ele)

            # print(len(GroundTruthNewList))
            DatasetObject.GroundTruth[key] = GroundTruthNewList

        end_creation_time = datetime.now().timestamp()
        creation_duration = end_creation_time - start_creation_time
        creation_date = str(datetime.now())

        DatasetObject.DatasetInfo = generate_dataset_info_filter(DatasetObject.DatasetInfo,
                                                                                      DatasetObject.Name,
                                                                                      DatasetObject.Name,
                                                                                      creation_date, creation_duration,
                                                                                      count, "minimum_bounding_box",
                                                              {"min_width": minimum_bounding_box_width, "min_hight": minimum_bounding_box_hight})

        return DatasetObject

    def export_to_json(self, path):
        """
        export Dataset info to   path as jsom info

        Input:
            path : Path where the results are exported

        Output:
            None

        """
        if not os.path.exists(path):
            os.mkdir(path)

        json_DatasetDict = os.path.join(path, "./DatasetDict.json")

        with open(json_DatasetDict, 'w') as fp:
            json.dump(self.DatasetDict, fp, indent=4)

        json_DatasetDict = os.path.join(path, "./GroundTruth.json")

        with open(json_DatasetDict, 'w') as fp:
            json.dump(self.GroundTruth, fp, indent=4)

        semantic_segmentation_DatasetDict = os.path.join(path, "./semantic_segmentation.json")

        with open(semantic_segmentation_DatasetDict, 'w') as fp:
            json.dump(self.semantic_segmentationDict, fp, indent=4)

        json_DatasetInfo = os.path.join(path, "./DatasetInfo.json")

        with open(json_DatasetInfo, 'w') as fp:
            json.dump(self.DatasetInfo, fp, indent=4)


    def export_to_coco_json(self, path):
        """
        export Dataset info to   path as COCO info

        Input:
            path : Path where the results are exported

        Output:
            None

        """

        if not os.path.exists(path):
            os.mkdir(path)

        images = []
        annotations = []
        categories = []
        categoriesDict = dict()

        categoriesId = 0
        annotations_id = 0

        for image_id, key in enumerate(self.GroundTruth.keys()):

            # print(key)
            filepath = self.DatasetDict.get(key)

            image = self.get(key)

            (im_width, im_height) = image.size

            images.append({"id": image_id,
                           "file_name": key,
                           "width": im_width,
                           "height": im_height,
                           "coco_url": filepath})

            for ele in self.get_Ground_truth(key):

                box = ele.get('box')
                Label = ele.get('Label')

                if Label != None and Label != "None":

                    category_id = categoriesDict.get(Label, None)

                    if category_id is None:
                        categoriesDict[Label] = categoriesId
                        category_id = categoriesId

                        categories.append({"id": category_id,
                                           "name": Label,
                                           "supercategory": "none"})

                        categoriesId = categoriesId + 1

                    Confidence = ele.get('Confidence', 1.0)

                    area = float(math.fabs((box[2] - box[0]) * (box[3] - box[1])))

                    annotations.append({"id": annotations_id,
                                        "segmentation": [],
                                        "area": area,
                                        "iscrowd": 0,
                                        "ignore": 0,
                                        "image_id": image_id,
                                        "bbox": [
                                            float(box[0]),
                                            float(box[1]),
                                            float(box[2]),
                                            float(box[3])
                                        ],
                                        "category_id": category_id,
                                        "score": Confidence})

                    annotations_id = annotations_id + 1

        resultDict = {"images": images, "annotations": annotations, "categories": categories}
        json_DatasetDict = os.path.join(path, "COCOJason.json")
        with open(json_DatasetDict, 'w') as fp:
            json.dump(resultDict, fp, indent=4)


def generate_dataset_info_subset(self, datasetInfo: dict, dataset_name, creation_date, creation_duration, count,
                                 dataset_name_source):
    # TODO
    return

def generate_dataset_info_filter(datasetInfo: dict, dataset_name,
                                 dataset_name_source,
                                 creation_date, creation_duration,
                                 count, filter_name, parameter):

    if "filter" not in datasetInfo:
        datasetInfo["filter"] = []
    datasetInfo["filter"].append({
        "dataset_name": dataset_name,
        "creation_duration": creation_duration,
        "creation_date": creation_date,
        "size": count,
        "source_dataset": dataset_name_source,
        "filter": {
            "name": filter_name,
            "parameter": parameter
        }
    })

    datasetInfo = generate_dataset_info_summary(datasetInfo, dataset_name, creation_date, count)
    # add short information about all used manipulations to the summary
    datasetInfo["summary"]["filter"].append({"name": filter_name, "parameter": parameter})

    return datasetInfo

def generate_dataset_info_manipulation(DatasetInfo: dict, DatasetName, creation_date, creation_duration,
                                       count, sourceDatasetName, manipulationName, value, manipulation_duration):
    """
        generate edge dataset

        Input:
            DatasetInfo : History of manipulations and information about the original dataset
            DatasetName: Name of the created dataset
            creation_date: Date of the creation
            generation_duration: Time needed to generate the dataset
            count: size of the dataset
            sourceDatasetName: Name of the source dataset
            manipulationName: Kind of manipulation used in the experiment
            value: strength of manipulation
            manipulation_duration: Time needed to manipulate the images

        Output:
            dictionary with information about the dataset and its history
    """

    # add the key for manipulations to the dataset if there is no one
    if "manipulations" not in DatasetInfo:
        DatasetInfo["manipulations"] = []

    # add a new manipulation step to the list of manipulations with information about the current manipulation
    DatasetInfo["manipulations"].append({
        "dataset_name": DatasetName,
        "creation_date": creation_date,
        "creation_duration": creation_duration,
        "size": count,
        "source_dataset": sourceDatasetName,
        "manipulation": {
            "name": manipulationName,
            "value": value,
            "duration": manipulation_duration
        }
    })

    DatasetInfo = generate_dataset_info_summary(DatasetInfo, DatasetName, creation_date, count)
    # add short information about all used manipulations to the summary
    DatasetInfo["summary"]["manipulations"].append({"name": manipulationName, "value": value})

    return DatasetInfo

def generate_dataset_info_summary(DatasetInfo: dict, DatasetName, creation_date, count):
    # add the key for summary to the dictionary if there is no one
    if "summary" not in DatasetInfo:
        DatasetInfo["summary"] = {}
        DatasetInfo["summary"]["dataset"] = {}
        DatasetInfo["summary"]["manipulations"] = []
        DatasetInfo["summary"]["filter"] = []
        DatasetInfo["summary"]["original_dataset"] = {}

    # add information about the new dataset to the summary
    DatasetInfo["summary"]["dataset"]["dataset_name"] = DatasetName
    DatasetInfo["summary"]["dataset"]["creation_date"] = creation_date
    DatasetInfo["summary"]["dataset"]["size"] = count

    # if there are information about a used subset, the variable "is_subset" will be True
    if "subsets" in DatasetInfo and len(DatasetInfo["subsets"]) > 0:
        DatasetInfo["summary"]["dataset"]["is_subset"] = True
    else:
        DatasetInfo["summary"]["dataset"]["is_subset"] = False

    # add information about the first used original dataset to the summary
    DatasetInfo["summary"]["original_dataset"]["dataset_name"] = DatasetInfo["original_dataset"]["dataset_name"]
    DatasetInfo["summary"]["original_dataset"]["is_synthetic"] = DatasetInfo["original_dataset"]["is_synthetic"]
    DatasetInfo["summary"]["original_dataset"]["creation_date"] = DatasetInfo["original_dataset"]["creation_date"]
    DatasetInfo["summary"]["original_dataset"]["size"] = DatasetInfo["original_dataset"]["size"]

    return DatasetInfo


