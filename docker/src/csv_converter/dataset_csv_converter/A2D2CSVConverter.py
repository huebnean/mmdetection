from csv_converter import definition
from csv_converter.dataset_csv_converter import DatasetCSVConverter
from kpi_calculation.precision_loss_new import get_orientation_angle

import math

class A2D2CSVConverter(DatasetCSVConverter):


        def calc_distance(self, object_info):
            """
                    function calculates the distance bins defined  in distance
                    and adds the information to the object gt
        
                    INPUT:
                        object_info: dictionary with the original ground-truth information for the current object
        
                    OUTPUT:
                        object_info: dictionary with the original ground-truth information of the current object extended by a value for its distance bin
        
            """
            if "center" in object_info:
                coords = object_info["center"]
                # square the axis distance values
                sqr_coords = [coord ** 2 for coord in coords]
                # calculate distance
                distance = math.sqrt(sum(sqr_coords))
        
                # find the correct distance bin
                idx = 0
                for dist in definition.A2D2_DISTANCE:
                    if distance < dist:
                        object_info["distance_bin"] = f"{definition.A2D2_DISTANCE[idx - 1]}_{dist}"
                        return object_info
                    idx += 1
        
                # add information if no distance bin could be calculated
                object_info["distance_bin"] = "no_distance_information"

            return object_info

        def calculate_orientation_bin(self, object_info):
            """
                    function calculates the orientation bins bin_0 - bin_7 (rotation around the y-axis)
                    and adds the information to the object gt

                    INPUT:
                        object_info: dictionary with the original ground-truth information for the current object

                    OUTPUT:
                        object_info: dictionary with the original ground-truth information of the current object extended by a value for its orientation bin

            """

            if "rot_angle" in object_info:
                orientation_angle = get_orientation_angle(object_info["rot_angle"], object_info["axis"],
                                                          object_info["center"])

                if orientation_angle > definition.A2D2_BINS[0][0] or orientation_angle <= definition.A2D2_BINS[0][1]:
                    orientation_bin = f"bin_0"
                else:
                    for idx, limit_value in reversed(list(enumerate(definition.A2D2_BINS))):
                        if limit_value[0] < orientation_angle <= limit_value[1]:
                            orientation_bin = f"bin_{idx}"
                            break

                object_info["orientation_bin"] = orientation_bin
            return object_info


        def get_label_row(self, dataset_object: dict, label_name: str):
            label_value = ""
            if label_name == "distance_bin":
                dataset_object = self.calc_distance(dataset_object)
                label_value = dataset_object['distance_bin']
            elif label_name == "orientation_bin":
                dataset_object = self.calculate_orientation_bin(dataset_object)
                label_value = dataset_object['orientation_bin']

            dataset_object_row = [label_name, label_value]

            return dataset_object_row
