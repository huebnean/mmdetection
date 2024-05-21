from csv_converter.kpi_csv_converter.KpiCSVConverter import KpiCSVConverter


class ConfidenceConverter(KpiCSVConverter):

    def get_kpi_result_rows(self, kpi: str, results: dict, model_name: str, classes: list):
        kpi_results_rows = []

        for image_name, objects in self.dataset.GroundTruth.items():

            if not objects:
                objects = []

            for object in objects:

                object_id = self.get_object_id(self.dataset_info[image_name], object["box"], classes)

                kpi_object_row = [self.dataset.Name, image_name, str(object_id), model_name]
                kpi_object_row.append(kpi)
                kpi_object_row.append(object["Confidence"])

                kpi_results_rows.append(kpi_object_row)

        return kpi_results_rows

if __name__ == '__main__':
    from data_preparation.datasets_interface import ImageDetectionDataset

    final_directory = ""
    dataset_name = "pandaset_dataset"

    dataset = ImageDetectionDataset.get_datasets_by_name(dataset_name)

    kpi_converter = ConfidenceConverter(final_directory, dataset)
    kpi_converter.export_kpi_to_csv("Confidence", None, "yolop-640-640.onnx", ["Car"])