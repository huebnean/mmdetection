import ast
import csv
import importlib
import os
import pandas as pd

from dataset import ImageDetectionDataset
import csv_converter.definition as definition
import utils.bertrandt_iou as bertrandt_iou


class KpiCSVConverter:

    def __init__(self, path: str, dataset: ImageDetectionDataset):
        self.path = path
        self.dataset = dataset

        self.dataset_info = self.get_dataset_info_from_csv()

    def read_dataset_csv(self):
        dataset_info = dict()

        # check if dataset info already in dataset.csv
        dataset_path = os.path.join(self.path, definition.CSV_DATASET_BB_FILE_NAME)
        if not os.path.exists(dataset_path):
            return dataset_info

        with open(dataset_path) as dataset_csv_file:
            dataset_csv_reader = csv.reader(dataset_csv_file, delimiter=',')

            for dataset_row in dataset_csv_reader:

                if dataset_row[0] == self.dataset.Name:

                    if dataset_row[1] not in dataset_info:
                        dataset_info[dataset_row[1]] = dict()

                    if int(dataset_row[2]) not in dataset_info[dataset_row[1]]:
                        dataset_info[dataset_row[1]][int(dataset_row[2])] = dict()

                    dataset_info[dataset_row[1]][int(dataset_row[2])] = {"box": dataset_row[3], "class": dataset_row[4]}

        return dataset_info


    def get_dataset_info_from_csv(self):
        dataset_info = self.read_dataset_csv()  # dict: image_name -> object_id -> object_info
        dataset_name = self.dataset.Name
        dataset_converter_name = definition.DATASET_NAME_TO_CONVERTER[dataset_name]

        dataset_converter = getattr(importlib.import_module(f"csv_converter.dataset_csv_converter.{dataset_converter_name}"), dataset_converter_name)(self.path)

        # write dataset info to dataset.csv
        if not dataset_info:
            dataset_converter.export_dataset_to_csv(dataset_name, self.dataset.GroundTruth)
            dataset_info = self.read_dataset_csv()
            return dataset_info

        # if the dataset.csv already contains the dataset, check if all images and objects are written
        data_to_write = dict()
        write_data = False
        for image_name, objects in self.dataset.GroundTruth.items():
            if image_name not in dataset_info:
                data_to_write[image_name] = objects
                dataset_info[image_name] = dict()
                for i in range(0, len(objects) - 1):
                    object_info = objects[i]
                    dataset_info[image_name][i] = {"box": object_info[definition.DATASET_BB_NAME[dataset_name]],
                                                   "class": object_info[definition.DATASET_CLASS_NAME[dataset_name]]}
            else:
                i = 1
                objects_to_append = {}
                for object_info in objects:
                    object_appended = False

                    for object_info_already_appended in dataset_info[image_name].values():
                        if str(object_info[definition.DATASET_BB_NAME[dataset_name]]) == object_info_already_appended["box"] and\
                                object_info[definition.DATASET_CLASS_NAME[dataset_name]].lower() == object_info_already_appended["class"].lower():
                            object_appended = True
                            break

                    if not object_appended:
                        j = max(dataset_info[image_name].keys()) + i

                        objects_to_append[str(j)] = {"box": object_info[definition.DATASET_BB_NAME[dataset_name]],
                                                       "class": object_info[definition.DATASET_CLASS_NAME[dataset_name]]}

                        if image_name not in data_to_write:
                            data_to_write[image_name] = dict()
                        data_to_write[image_name][j] = object_info

                        write_data = True
                        i += 1

        if write_data:
            dataset_converter.export_dataset_to_csv(dataset_name, data_to_write)
            dataset_info = self.read_dataset_csv()

        return dataset_info


    def get_object_info_for_image(self, dataset_name: str, image_name: str, cls: str):

        return None


    def get_object_id(self, dict_of_objects: dict, bounding_box: list, cls: list):
        """
               function that maps the object ids for an image based on the dataset csv
               TODO: How to match objects of a dataset without ground truth for comparing different KPIs???

               INPUT:
                   dict_of_objects: dictionary of objects for the image from the dataset csv (key: object_id, value: {box, class})
                   bounding_box: [x1, y1, x2, y2]

               OUTPUT:
                   object_id: matching object id
        """

        max_iou = 0
        object_id = None

        for i, ref_obj in dict_of_objects.items():
            if ref_obj["class"] in cls:
                ref_obj_bb = ast.literal_eval(ref_obj["box"])
                iou = bertrandt_iou.intersection_over_union(bounding_box, ref_obj_bb)
                if iou > max_iou:
                    max_iou = iou
                    object_id = i

        return object_id


    def export_kpi_to_csv(self, kpi: str, results: dict, model_name: str, classes: list):

        csv_kpi_path = os.path.join(self.path, definition.CSV_KPI_FILE_NAME_PREFIX + "_" + kpi + ".csv")
        csv_kpi_file = open(csv_kpi_path, 'a', newline='')
        csv_kpi_file_writer = csv.writer(csv_kpi_file)
        try:
            pd.read_csv(csv_kpi_path, nrows=1)
        except:
            csv_kpi_file_writer.writerow(definition.KPI_FIRST_ROW)

        result_rows = self.get_kpi_result_rows(kpi, results, model_name, classes)
        for row in result_rows:
            csv_kpi_file_writer.writerow(row)

        csv_kpi_file.close()


    def get_kpi_result_rows(self, kpi: str, results: dict, model_name: str):

        pass





if __name__ == '__main__':
    bdd100k_dataset = ImageDetectionDataset.get_datasets_by_name("bdd100k_dataset")
    kpi_converter = KpiCSVConverter(r"C:\Users\huebnean\OneDrive - Bertrandt AG\Desktop", bdd100k_dataset)