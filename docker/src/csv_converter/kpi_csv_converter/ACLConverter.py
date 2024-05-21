from csv_converter.kpi_csv_converter.KpiCSVConverter import KpiCSVConverter


class ACLConverter(KpiCSVConverter):

    def get_kpi_result_rows(self, kpi: str, results: dict, model_name: str, classes: list):
        kpi_results_rows = []

        if results is None:
            return kpi_results_rows
            
        for kpi_name, kpi_values in results.items():
            print(self.dataset.Name)
            for image_name, objects in self.dataset.GroundTruth.items():

                for i in range(0, len(objects)):

                    for kpi_value, kpi_value_results in kpi_values.items():
                        res = kpi_value_results["conf_loss_list_dict"]

                        if image_name in res and i < len(res[image_name]):
                            kpi_res = res[image_name][i]["conf_loss_value"]
                        else:
                            continue

                        object_id = self.get_object_id(self.dataset_info[image_name], res[image_name][i]["ref_box"], classes)

                        kpi_object_row = [self.dataset.Name, image_name, str(object_id), model_name]
                        kpi_object_row.append(kpi + "_" + kpi_name + "_" + kpi_value)
                        kpi_object_row.append(kpi_res)

                        kpi_results_rows.append(kpi_object_row)

        return kpi_results_rows