import csv
import os
import pandas as pd

from csv_converter import definition
import utils.bertrandt_iou as iou

class DatasetCSVConverter:

    def __init__(self, path):
        self.path = path

    def get_object_id(self, dict_of_objects: dict, bounding_box: list, cls: str):
        """
               function that maps the object ids for an image based on the dataset csv
      CSV_DATASET_LABEL_FILE_NAME         TODO: How to match objects of a dataset without ground truth for comparing different KPIs???

               INPUT:
                   dict_of_objects: dictionary of objects for the image from the dataset csv (key: object_id, value: {box, class})
                   bounding_box: [x1, y1, x2, y2]

               OUTPUT:
                   object_id: matching object id
        """

        object_id = None

        ref_ious = [iou.intersection_over_union(bounding_box,
                                                ref_obj["box"]) for ref_obj in dict_of_objects if
                    ref_obj["class"] is cls]

        if len(ref_ious) > 0:
            object_id = ref_ious.index(max(ref_ious))
        else:
            print("KPI result contains an object without ground truth.")

        return object_id


    def export_dataset_to_csv(self, dataset_name: str, dataset_info: dict):
        csv_dataset_bb_path = os.path.join(self.path, definition.CSV_DATASET_BB_FILE_NAME)
        csv_dataset_bb_file = open(csv_dataset_bb_path, 'a', newline='')
        csv_dataset_bb_file_writer = csv.writer(csv_dataset_bb_file)
        try:
            pd.read_csv(csv_dataset_bb_path, nrows=1)
        except:
            csv_dataset_bb_file_writer.writerow(definition.DATASET_BB_FIRST_ROW)

        csv_dataset_label_path = os.path.join(self.path, definition.CSV_DATASET_LABEL_FILE_NAME)
        csv_dataset_label_file = open(csv_dataset_label_path, 'a', newline='')
        csv_dataset_label_file_writer = csv.writer(csv_dataset_label_file)
        try:
            pd.read_csv(csv_dataset_label_path, nrows=1)
        except:
            csv_dataset_label_file_writer.writerow(definition.DATASET_LABEL_FIRST_ROW)

        for image_name, objects in dataset_info.items():
            if objects.__class__ is dict:
                max_i = max(objects.keys()) + 1
            else:
                max_i = len(objects)

            for i in range(0, max_i):

                if i not in objects and objects.__class__ is dict:
                    continue

                dataset_object = objects[i]

                if not dataset_object:
                    continue

                bb_original_information = dataset_object[definition.DATASET_BB_NAME[dataset_name]]
                center_x, center_y, width, height = self.get_bb_information(bb_original_information)

                dataset_object_bb_row = [dataset_name, image_name, str(i), bb_original_information,
                                          dataset_object[definition.DATASET_CLASS_NAME[dataset_name]],
                                         center_x, center_y, width, height]
                csv_dataset_bb_file_writer.writerow(dataset_object_bb_row)

                for label_name in definition.LABEL_NAMES[dataset_name]:
                    dataset_object_label_row = [dataset_name, image_name, str(i)] + self.get_label_row(dataset_object, label_name)
                    csv_dataset_label_file_writer.writerow(dataset_object_label_row)

        csv_dataset_bb_file.close()
        csv_dataset_label_file.close()


    def get_label_row(self, dataset_object: dict, label_name: str):
        pass

    def get_bb_information(self, bb_original_information):
        pass
