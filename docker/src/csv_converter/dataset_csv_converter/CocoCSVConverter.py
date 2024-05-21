from csv_converter.dataset_csv_converter.DatasetCSVConverter import DatasetCSVConverter


class CocoCSVConverter(DatasetCSVConverter):

    def get_label_row(self, dataset_object: dict, label_name: str):
        return [label_name, dataset_object[label_name]]

    def get_bb_information(self, bb_original_information):
        # original bounding box -> list: [x1, y1, x2, y2]
        width = bb_original_information[2] - bb_original_information[0]
        height = bb_original_information[3] - bb_original_information[1]

        center_x = bb_original_information[0] + (width / 2)
        center_y = bb_original_information[1] + (height / 2)

        return center_x, center_y, width, height