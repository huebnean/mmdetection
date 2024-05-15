"""


"""
import math
import os,json
import shutil
import numpy as np
import scipy.stats as st

import utils.bertrandt_iou as iou
from dataset import ImageDetectionDataset
from csv_converter.kpi_csv_converter.ACLConverter import ACLConverter

# list of all filter keys that do not need a second dataset for the reference objects
FILTER_KEYS_SAME_DATASET = ['color', 'distance_bin', 'source', 'gan_idx', 'orientation_bin']
# list of all filter keys that need a second dataset for the reference objects
FILTER_KEYS_DIFF_DATASET = ['motion_blur', 'blur', 'brightness', 'gaussian_noise', 'flip', 'mirror', 'rotate',
                            'saturation', 'dropout', 'contrast', 'illumination']
# list of all filter keys that need to handle different bounding boxes for the same object
FILTER_KEYS_DIFF_DATASET_CHANGED_BB = ['flip', 'mirror', 'rotate']
# list of labels
CLASSES_TO_COMPARE = ['car', 'pedestrian']
# list of bins for object rotation around the y-axis
BINS = [
    (5.891, 0.393),  # ]337.5°, 22.5°]
    (0.393, 1.178),  # ]22.5°, 67.5°]
    (1.178, 1.964),  # ]67.5°, 112.5°]
    (1.964, 2.749),  # ]112.5°, 157.5°]
    (2.749, 3.534),  # ]157.5°, 202.5°]
    (3.534, 4.320),  # ]202.5°, 247.5°]
    (4.320, 5.105),  # ]247.5°, 292.5°]
    (5.105, 5.891)  # ]292.5°, 337.5°]
]
# list for distance of the object
DISTANCE = [0, 10, 20, 30, 40, 50]


def calc_distance(object_info):
    """
            function calculates the distance bins defined  in DISTANCE
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
        for dist in DISTANCE:
            if distance < dist:
                object_info["distance_bin"] = f"{DISTANCE[idx - 1]}_{dist}"
                return object_info
            idx += 1

        # add information if no distance bin could be calculated
        object_info["distance_bin"] = "no_distance_information"
        return object_info


def calculate_orientation_bin(object_info):
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

        if orientation_angle > BINS[0][0] or orientation_angle <= BINS[0][1]:
            orientation_bin = f"bin_0"
        else:
            for idx, limit_value in reversed(list(enumerate(BINS))):
                if limit_value[0] < orientation_angle <= limit_value[1]:
                    orientation_bin = f"bin_{idx}"
                    break

        object_info["orientation_bin"] = orientation_bin
    return object_info


def filter_to_manipulated_datasets(filter: dict, reference_dataset: ImageDetectionDataset,
                                   manipulated_datasets_infer: list, filter_keys_same: list, filter_keys_diff: list,
                                   filter_keys_diff_changed_bb: list, object_source: list):
    """
            This function maps each filter listed in the config file to the matching manipulated dataset and dds some important
            information like the orientation bin. This function checks also if there are any problems with the defined filters
            and the datasets. If a problem is detected, the function is throwing an error with helpful information.

            INPUT:
                filter: dictionary with the filter names and their values (e.g. blur -> 1, 2, 3, ...) \n
                reference_dataset: the dataset which is used as reference for the calculation of the ACL \n
                manipulated_datasets_infer: list with names of the manipulated datasets \n
                filter_keys_same: list with all filter keys that do not need a second dataset for the reference objects (e.g. color) \n
                filter_keys_diff: list with all filter keys that need a second dataset for the reference objects (e.g. blur) \n
                filter_keys_diff_changed_bb: list with all filter keys that need to handle different bounding boxes for the same object (e.g. rotation) \n

            OUTPUT:
                filter_to_manipulated_dataset: dictionary with key for each filter and the matching manipulated dataset as their values \n
                filter_diff_dataset: dictionary with keys for the filter names and their values for filter_keys_diff \n
                filter_same_dataset: dictionary with keys for the filter names and their values for filter_keys_same \n
                filter_to_ref_eval: dictionary with keys for the filter names and values and the matching images for each value \n

    """

    filter_same_dataset = {}
    filter_diff_dataset = {}
    filter_to_manipulated_dataset = {}
    filter_to_ref_eval = {}

    filter_valid = True

    # loop over all defined filters
    for filter_name, filter_values in filter.items():

        # separate the filter values into a list of strings if there are one
        try:
            if isinstance(filter_values, list):
                filter_values = filter_values
                if isinstance(filter_values[0], dict):
                    filter_values_list = []
                    for filter_value in filter_values:
                        filter_values_list.append(str(filter_value))
                    filter_values = filter_values_list

            elif isinstance(filter_values, dict):
                filter_values = filter_values
            else:
                filter_values = [split.strip() for split in filter_values.split(",")]
        except:
            filter_values = [str(filter_values)]

        # if current filter needs a ref dataset and a different manipulated dataset
        if filter_name in filter_keys_diff:

            # check if there are manipulated datasets in the config file
            if len(manipulated_datasets_infer) > 0:

                # add current filter to dictionaries
                filter_diff_dataset[filter_name] = filter_values
                filter_to_manipulated_dataset[filter_name] = {}

                # copy of all filter values of the current filter
                filter_values_copy = filter_values.copy()

                # Loop over all manipulated datasets
                for manipulated_dataset_infer in manipulated_datasets_infer:
                    dataset_name = manipulated_dataset_infer.DatasetInfo["summary"]["dataset"]["dataset_name"]

                    # check if manipulation datasets are containing the same images
                    for manipulated_dataset_infer_2 in manipulated_datasets_infer:
                        dataset_name_2 = manipulated_dataset_infer_2.DatasetInfo["summary"]["dataset"]["dataset_name"]

                        if manipulated_dataset_infer_2 != manipulated_dataset_infer and \
                                (len(manipulated_dataset_infer_2.GroundTruth) != len(
                                    manipulated_dataset_infer.GroundTruth) or
                                 manipulated_dataset_infer_2.GroundTruth.keys() != manipulated_dataset_infer.GroundTruth.keys()):
                            raise DatasetException(
                                f"Dataset {dataset_name} and {dataset_name_2} do not contain the same images.")

                    # copy of all filter names
                    filter_copy = filter.keys()

                    # TODO: For filter ROTATE, check if the max angle is the same for all datasets (image sizes must be the same to compare the ACLs)

                    # Loop over all manipulations of the manipulated dataset to find the right one
                    for manipulation in manipulated_dataset_infer.DatasetInfo["manipulations"]:
                        manipulation_name = manipulation["manipulation"]["name"]
                        manipulation_value = manipulation["manipulation"]["value"]
                        manipulation_value_str = str(manipulation_value)

                        # check if manipulation is duplicated in the current manipulated dataset
                        if manipulation_name in filter and manipulation_name not in filter_copy:
                            raise DatasetException(
                                f"Manipulation {manipulation_name} is duplicated in {dataset_name}")

                        # check if dataset contains a manipulation which is not defined in the filter
                        if manipulation_name not in filter:
                            raise DatasetException(
                                f"Manipulation {manipulation_name} not specified in filter list.")

                        # check if manipulation value is a filter value
                        if manipulation_name.lower() == filter_name.lower() and manipulation_value_str in filter_values:
                            # check if manipulated datasets contains the same filter value
                            if manipulation_value_str in filter_values_copy:
                                if filter_name in filter_keys_diff_changed_bb:
                                    try:
                                        manipulated_dataset = ImageDetectionDataset.get_datasets_by_name(dataset_name)
                                        filter_to_manipulated_dataset[filter_name][manipulation_value_str] = \
                                            {'dataset': manipulated_dataset, 'infer': manipulated_dataset_infer}

                                    except:
                                        raise DatasetException(f"No dataset {dataset_name} found.")
                                else:
                                    filter_to_manipulated_dataset[filter_name][manipulation_value_str] = \
                                        {'dataset': dataset_name, 'infer': manipulated_dataset_infer}

                                filter_values_copy.remove(manipulation_value_str)

                            elif manipulation_value_str in filter_values:
                                raise DatasetException(
                                    f"Manipulation {manipulation_name} {manipulation_value_str} is duplicated.")



                # check if there was a manipulated dataset with the manipulation defined in the current filter
                if filter_to_manipulated_dataset[filter_name] == {}:
                    print(f"No manipulated dataset contains a manipulation matching the filter {filter_name}")

                # check if all values for the defined filter contained in manipulated datasets
                if filter_values_copy:
                    print(
                        f"No manipulated dataset contains a manipulation matching the filter value {filter_values_copy}")

            else:
                raise DatasetException("Filter for different datasets can not work without a manipulated dataset.")

        # if current filter uses ref and eval images from the same dataset (e.g. color)
        elif filter_name in filter_keys_same:
            filter_same_dataset[filter_name] = filter_values
            filter_to_ref_eval[filter_name] = {}

            # add all defined filter values for eval objects to the dictionary filter_to_ref_eval
            for eval in filter_values["eval"]:
                filter_to_ref_eval[filter_name][eval.lower()] = {}

            # add the defined reference to filter_to_ref_eval
            filter_to_ref_eval[filter_name][filter_values["ref"]] = {}

            # loop over all images the reference dataset contains for sorting each object to the right filter value
            for image, image_info in reference_dataset.GroundTruth.items():

                filter_in_dataset = False

                # loop over all objects the image contains
                for box in image_info:
                    #if "source" not in box or box["source"] not in object_source:
                    #    continue

                    # add bin information
                    if filter_name == "orientation_bin":
                        box = calculate_orientation_bin(box)

                    if filter_name == "distance_bin":
                        box = calc_distance(box)

                    # add correct color information
                    if filter_name == "color":
                        box["color"] = box["foreground_object_color"]

                    # ignore object if it has no information for the current filter
                    # or the value of the object for the current filter is not specified in the config file
                    if filter_name not in box or box[filter_name].lower() not in filter_to_ref_eval[filter_name]:
                        continue

                    # add a list for current image to dictionary filter_to_ref_eval
                    if image not in filter_to_ref_eval[filter_name][box[filter_name].lower()]:
                        filter_to_ref_eval[filter_name][box[filter_name].lower()][image] = []

                    # append the current box to the list
                    filter_to_ref_eval[filter_name][box[filter_name].lower()][image].append(box)
                    filter_in_dataset = True

                if not filter_in_dataset:
                    print(f"WARNING: Filter {filter_name} not in Dataset: {reference_dataset.Name}")
        else:
            raise DatasetException("Filter is not supported: " + filter_name)

    return filter_to_manipulated_dataset, filter_diff_dataset, filter_same_dataset, filter_to_ref_eval


def add_conf_loss_same_dataset(conf_loss_filter_dict, filter_name_final, filter_value, image_name,
                               reference_dataset_infer,
                               ref_image, eval_image_infer,
                               iou_threshold, ignore_conf_gain, ignore_no_detection, object_source, min_bb, use_gt,
                               filter_name):
    """
            function adds the ACL for a pair of ref and eval object to the dictionary which contains all ACLs for each object and filter
            only for case if ref and eval object in the same dataset (e.g. color)

            INPUT:
                conf_loss_filter_dict: dictionary which contains all ACLs for each object and filter \n
                filter_name: current filter name \n
                filter_value: current filter value \n
                image_name: name of the eval image \n
                reference_dataset_infer: dataset which contains all confidence scores for the reference objects \n
                ref_image: reference image inference based on image name and predicted BBs and confidence scores \n
                eval_image_infer: eval image inference based on image name and predicted BBs and confidence scores\n
                iou_threshold: threshold to define the minimum IoU the inferences and the ground truth objects have to have for mapping them\n
                ignore_conf_gain: if true, higher eval confidence score will be ignored \n
                ignore_no_detection: if true, no detections will not be apart of the ACL calculation \n
                object_source: list of all possible object sources (e.g. synthetic_gan, a2d2) \n
                min_bb: min size of width and hight  \n


            OUTPUT:
                conf_loss_filter_dict: dictionary with all ACLs for each pair of ref and eval object separated by filters

    """
    # ignore and print some warning for unusual situations
    if len(ref_image) == 0:
        print(f"WARNING: no reference image for {image_name}")
    elif len(ref_image) > 1:
        print(f"WARNING: More than one reference image for {image_name}")

    # if no unusual situation:
    else:

        # get the ground truth
        image_ground_truth = list(ref_image[0].values())[0]
        ref_image_infer = reference_dataset_infer.GroundTruth[list(ref_image[0].keys())[0]]

        # compare ref and eval image
        images_compare_results = compareObjects(image_name, image_ground_truth, eval_image_infer,
                                                ref_image_infer, None,
                                                iou_threshold, ignore_conf_gain,
                                                ignore_no_detection,
                                                object_source, min_bb, filter_name, filter_value, use_gt)

        # add the results to the conf_loss_filter_dict
        if len(images_compare_results) > 0:
            conf_loss_dict = {
                "image_name": image_name,
                "compared_objects": images_compare_results
            }

            if filter_value not in conf_loss_filter_dict[filter_name_final]:
                conf_loss_filter_dict[filter_name_final][filter_value] = []
            conf_loss_filter_dict[filter_name_final][filter_value].append(conf_loss_dict)

    return conf_loss_filter_dict


def compute_metric(reference_dataset: ImageDetectionDataset, reference_dataset_infer: ImageDetectionDataset,
                   manipulated_datasets_infer: list, filter=None,
                   iou_threshold: float = 0.5, ignore_conf_gain: bool = True,
                   ignore_no_detection: bool = False, object_source: str = "", min_bb=None, use_gt: bool = False, result_path = str):
    """
            main function that generate a dictionary of ACLs for a given set of filters, an original dataset and manipulated datasets

            INPUT:
                reference_dataset: ImageDetectionDataset object of the reference dataset \n
                reference_dataset_infer: ImageDetectionDataset object of the inference of reference dataset \n
                manipulated_datasets_infer: list of ImageDetectionDataset objects of the inferences of manipulated datasets \n
                filter: dictionary of all defined filters (names and values) \n
                iou_threshold: IoU threshold for matching the inference objects to the GT-objects \n
                ignore_conf_gain: if true, higher eval confidence score will be ignored \n
                ignore_no_detection: if true, no detections will not be apart of the ACL calculation \n
                object_source: list of all possible object sources (e.g. synthetic_gan, a2d2) \n
                min_bb: min size of width and hight  \n

            OUTPUT:
                conf_loss_filter_dict_result: dictionary with all ACLs for each pair of ref and eval object separated by filters

    """
    if os.path.exists(result_path):
        print("WARNING Path already exists: ", result_path)

    result_path_subs = os.path.join(result_path, "acl_results")

    # get list of possible sources (e.g. a2d2, synthetic_gan...)
    if filter is None:
        filter = {}
    if min_bb is None:
        min_bb = [0, 0]
    if object_source is not None:
        object_source = [split.strip() for split in object_source.split(",")]

    conf_loss_filter_dict = {}  # filter_name -> filter_value -> [ compared_objects ]
    conf_loss_filter_dict_result = {}  # filter_name -> filter_value -> { [ compared_objects ], mean, std }

    filter_to_manipulated_dataset = {}  # filter_name -> filter_value -> manipulated_dataset
    filter_diff_dataset = {}  # filter_name -> [ filter_values ]
    filter_same_dataset = {}  # filter_name -> [ filter_values ]
    filter_to_ref_eval = {}  # filter_name -> filter_value -> image

    # try to connect all defined filter and their values to the datasets
    # and check if there are some problems with the filter or the datasets
    try:
        filter_to_manipulated_dataset, filter_diff_dataset, filter_same_dataset, filter_to_ref_eval = filter_to_manipulated_datasets(
            filter, reference_dataset, manipulated_datasets_infer, FILTER_KEYS_SAME_DATASET, FILTER_KEYS_DIFF_DATASET,
            FILTER_KEYS_DIFF_DATASET_CHANGED_BB, object_source)
    except DatasetException as error:
        print(f"Error while connecting filter of different datasets to manipulated datasets: {error}")

    # counter for compared images
    count = 0
    diff_count = 0
    count_all_images = len(reference_dataset.GroundTruth.keys())

    # Loop over all objects the reference dataset contains
    for image_name, image_ground_truth in reference_dataset.GroundTruth.items():
        count = count + 1
        diff_count += 1

        #if diff_count > 110:
            #break

        if diff_count % 100 == 0:
            print("wrote to disc")
            #directory = os.path.dirname(result_path_subs)
            final_directory = result_path_subs
            i = 0
            while os.path.exists(final_directory):
                final_directory = result_path_subs + f"_{i}"
                i += 1

            os.makedirs(final_directory)

            with open(os.path.join(final_directory, 'result_detailed.json'), 'w') as fp:
                json.dump(conf_loss_filter_dict, fp)

            conf_loss_filter_dict = {}

        # print warning if no inference exists for the current image
        if image_name in reference_dataset_infer.GroundTruth:
            original_image_infer = reference_dataset_infer.GroundTruth[image_name]
        else:
            print(f"{count} of {count_all_images}: No inference for {image_name}")
            continue

        # Loop over all filter for same datasets
        for filter_name, filter_values in filter_same_dataset.items():

            # add new dictionaries for current filter
            if filter_name not in conf_loss_filter_dict:
                conf_loss_filter_dict[filter_name] = {}
                conf_loss_filter_dict_result[filter_name] = {}

            # get reference filter value
            ref = filter_values["ref"]

            # ignore the image if no reference defined or the image itself contains the reference object
            if image_name in filter_to_ref_eval[filter_name][ref] or ref is None:
                continue

            # get list of evals
            evals = filter_values["eval"]

            # counter for objects with the current filter the image contains
            image_has_filter = 0

            # loop over all values defined as evals
            filter_value_evals = []
            eval_image_infer = None

            for eval in evals:

                # find the current filter_value based on the object which contains the filter_name
                if image_name in filter_to_ref_eval[filter_name][eval]:
                    eval_image_infer = reference_dataset_infer.GroundTruth[image_name]
                    filter_value_evals.append(eval)
                    image_has_filter += 1

            print(
                f'{count} of {count_all_images}, {filter_name}: comparing {image_has_filter} objects of {image_name} from {reference_dataset.Name}')

            # get the prefix of image name to find its reference image
            image_prefix = image_name[0: image_name.rfind("_") + 1]
            ref_image = [{key: val} for key, val in filter_to_ref_eval[filter_name][ref].items() if
                         key.startswith(image_prefix)]

            # add ACLs for ref and eval objects to conf_loss_filter_dict
            for filter_value_eval in filter_value_evals:
                conf_loss_filter_dict = add_conf_loss_same_dataset(conf_loss_filter_dict, filter_name,
                                                                   filter_value_eval, image_name,
                                                                   reference_dataset_infer, ref_image, eval_image_infer,
                                                                   iou_threshold, ignore_conf_gain, ignore_no_detection,
                                                                   object_source,
                                                                   min_bb, use_gt, filter_name)

        # loop over all filter for different dataset (e.g. rotation, blur...)
        for filter_name, filter_values in filter_diff_dataset.items():

            # check if the current image is part of manipulated datasets
            if image_name in manipulated_datasets_infer[0].keys():

                # ignore current filter if there are no dataset containing it
                if filter_name not in filter_to_manipulated_dataset:
                    continue

                # add new dictionaries for the current filter
                if filter_name not in conf_loss_filter_dict:
                    conf_loss_filter_dict[filter_name] = {}
                    conf_loss_filter_dict_result[filter_name] = {}

                # Loop over all filter values
                for filter_value in filter_values:
                    str_filter_value = str(filter_value)

                    # TODO: find better solution
                    # add new list for current filter value
                    if str_filter_value not in str(conf_loss_filter_dict[filter_name]) or str_filter_value not in \
                            conf_loss_filter_dict[filter_name].keys():
                        conf_loss_filter_dict[filter_name][str_filter_value] = []

                    # get the inference of manipulations for current filter and filter value
                    manipulated_dataset_infer = filter_to_manipulated_dataset[filter_name][str_filter_value]['infer']
                    manipulated_image_infer = manipulated_dataset_infer.GroundTruth[image_name]

                    # add manipulated Ground Truth if the filter needs different Ground Truth BBs for manipulation (e.g. rotation)
                    if filter_name in FILTER_KEYS_DIFF_DATASET_CHANGED_BB:
                        manipulated_dataset = filter_to_manipulated_dataset[filter_name][str_filter_value]['dataset']
                        manipulated_image_gt = manipulated_dataset.GroundTruth[image_name]
                    else:
                        manipulated_image_gt = None

                    print(
                        f'{count} of {count_all_images}, {filter_name}: comparing {image_name} from {reference_dataset.Name} to {manipulated_dataset_infer.Name}')

                    # compare ref and eval images
                    images_compare_results = compareObjects(image_name, image_ground_truth, manipulated_image_infer,
                                                            original_image_infer, manipulated_image_gt, iou_threshold,
                                                            ignore_conf_gain,
                                                            ignore_no_detection, object_source, min_bb, filter_name,
                                                            filter_value, use_gt)

                    # add results to conf_loss_filter_dict
                    if len(images_compare_results) > 0:
                        conf_loss_dict = {
                            "image_name": image_name,
                            "compared_objects": images_compare_results
                        }

                        conf_loss_filter_dict[filter_name][str_filter_value].append(conf_loss_dict)

                    # filter for only one dataset for each manipulated dataset
                    # TODO: usability without GT
                    for filter_name_2, filter_values_2 in filter_same_dataset.items():

                        # get new filter name
                        # composed of the filter for the manipulated datasets and the filter for same dataset
                        connected_filter_name = filter_name + "-" + str_filter_value + "-" + filter_name_2

                        # add new dictionaries for the filter
                        if connected_filter_name not in conf_loss_filter_dict:
                            conf_loss_filter_dict[connected_filter_name] = {}
                            conf_loss_filter_dict_result[connected_filter_name] = {}

                        # get the reference filter value
                        ref = filter_values_2["ref"]

                        # ignore filter if reference is defined and the current image contains the reference object
                        if ref is not None and image_name in filter_to_ref_eval[filter_name_2][ref]:
                            continue

                        # loop over all values defined as evals
                        evals = filter_values_2["eval"]
                        image_has_filter = 0
                        filter_value_evals = []
                        eval_image_infer = None
                        for eval in evals:

                            # find the current filter_value based on the object which contains the filter_name
                            if image_name in filter_to_ref_eval[filter_name_2][eval]:
                                eval_image_infer = manipulated_image_infer
                                filter_value_evals.append(eval)
                                image_has_filter += 1

                        print(
                            f'{count} of {count_all_images}, {connected_filter_name}: comparing {image_has_filter} objects of {image_name} from {reference_dataset.Name}')

                        # get the prefix of image name to find its reference image
                        image_prefix = image_name[0: image_name.rfind("_") + 1]

                        # if no reference specified: use the original reference image
                        # else: find the reference image based on prefix (reference inference is always from the original reference)
                        if ref is None:
                            ref_image = [{image_name: reference_dataset.GroundTruth[image_name]}]
                        else:
                            ref_image = [{key: val} for key, val in filter_to_ref_eval[filter_name_2][ref].items() if
                                         key.startswith(image_prefix)]

                        # add ACLs to conf_loss_filter_dict
                        for filter_value_eval in filter_value_evals:
                            conf_loss_filter_dict = add_conf_loss_same_dataset(conf_loss_filter_dict,
                                                                               connected_filter_name,
                                                                               filter_value_eval, image_name,
                                                                               reference_dataset_infer, ref_image,
                                                                               eval_image_infer,
                                                                               iou_threshold, ignore_conf_gain,
                                                                               ignore_no_detection, object_source,
                                                                               min_bb, use_gt, filter_name_2)


            else:
                print(f'WARNING: {image_name} is not present in any manipulated dataset')

    directory = result_path_subs
    dir_to_sub_result = directory
    i = 0
    while os.path.exists(dir_to_sub_result):
        dir_to_sub_result = directory + f"_{i}"
        i += 1

    os.makedirs(dir_to_sub_result)

    with open(os.path.join(dir_to_sub_result, 'result_detailed.json'), 'w') as fp:
        json.dump(conf_loss_filter_dict, fp)


    conf_loss_filter_dict = collect_results(result_path_subs)
    conf_loss_filter_dict_result = {}

    # loop over all filter and their values and calculate mean ACLs and their stds
    for filter_name, filter_values in conf_loss_filter_dict.items():
        if filter_name not in conf_loss_filter_dict_result:
            conf_loss_filter_dict_result[filter_name] = {}
        for filter_value in filter_values:
            str_filter_value = str(filter_value)
            print(f"Mean and std for {filter_name} {filter_value}")

            # ToDo: Different calculation for the filter color -> std for each unique object -> mean of stds = new std for current color
            conf_loss_filter_dict_result[filter_name][str_filter_value] = output_mean(
                conf_loss_filter_dict[filter_name][str_filter_value])


    directory = os.path.dirname(result_path)
    final_directory = os.path.join(directory, 'final_results')

    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    with open(os.path.join(final_directory, 'result_detailed.json'), 'w') as fp:
        json.dump(conf_loss_filter_dict_result, fp)

    result_summary = get_result_summary(conf_loss_filter_dict_result)

    with open(os.path.join(final_directory, 'result_summary.json'), 'w') as fp:
        json.dump(result_summary, fp)

    model_name = os.path.basename(os.path.normpath(reference_dataset_infer.DatasetInfo["inference"]["model"]["path"]))
    save_results_as_csv(conf_loss_filter_dict_result, reference_dataset, model_name, ["car"], final_directory)

    return conf_loss_filter_dict_result


def compareObjects(image_name, image_ground_truth, eval_img, ref_img, manipulated_image_gt, iou_threshold,
                   ignore_conf_gain,
                   ignore_no_detection, object_source, min_bb, filter_name, filter_value, use_gt):
    """
            function compares the eval and ref images and calculates the ACLs for each object occurring in the gt image

            INPUT:
                image_name: name of the current image \n
                image_ground_truth: list of all Ground Truth BBs for each object occurring in the current image \n
                eval_img: inference of eval image \n
                ref_img: inference of ref image \n
                manipulated_image_gt: Ground Truth for manipulated eval image (necessary if positions changed, e.g. rotation)
                iou_threshold: IoU threshold for matching the inference objects to the GT-objects \n
                ignore_conf_gain: if true, higher eval confidence score will be ignored \n
                ignore_no_detection: if true, no detections will not be apart of the ACL calculation \n
                object_source: list of all possible object sources (e.g. synthetic_gan, a2d2) \n
                min_bb: min size of width and hight  \n

            OUTPUT:
                conf_loss_list_img: list containing all objects and its ACL, confidence scores for ref and eval inference and other information

    """

    conf_loss_list_img = []

    # get all comparable objects based on the Ground Truth
    if use_gt:
        objects_to_compare = [obj for obj in image_ground_truth if
                              obj.get("class").lower() in CLASSES_TO_COMPARE and ("source" not in obj or
                              obj.get("source") in object_source) and
                              (filter_name in FILTER_KEYS_SAME_DATASET and obj.get(filter_name) == filter_value or
                               filter_name in FILTER_KEYS_DIFF_DATASET) and
                              obj.get("box")[2] - obj.get("box")[0] >= min_bb[0] and
                              obj.get("box")[3] - obj.get("box")[1] >= min_bb[1]]

    # get all comparable objects based on the inference of the reference dataset
    else:
        objects_to_compare = [obj for obj in ref_img if
                              obj.get("class").lower() in CLASSES_TO_COMPARE and
                              (filter_name in FILTER_KEYS_SAME_DATASET and obj.get(filter_name) == filter_value or
                               filter_name in FILTER_KEYS_DIFF_DATASET) and
                              obj.get("box")[2] - obj.get("box")[0] >= min_bb[0] and
                              obj.get("box")[3] - obj.get("box")[1] >= min_bb[1]]

    # loop over all comparable objects
    for object in objects_to_compare:

        # TODO: rotation etc without GT
        # get the ground truth for manipulated image if necessary (e.g. rotation, flip..)
        if manipulated_image_gt is None:
            object_eval = object
        else:
            no_match = True
            counter = 0
            while no_match and len(manipulated_image_gt) > counter:
                no_match = manipulated_image_gt[counter]['box'] != object['box']
                object_eval = manipulated_image_gt[counter]
                counter += 1
            if no_match:
                raise DatasetException(f'No GT-Value for manipulated dataset for image {image_name}')

        # get ious for the inference and ground truth
        ref_ious = [iou.intersection_over_union(object["box"],
                                            ref_obj["box"]) for ref_obj in ref_img if
                        ref_obj["class"].lower() in CLASSES_TO_COMPARE]
        eval_ious = [iou.intersection_over_union(object_eval["box"],
                                                 eval_obj["box"]) for eval_obj in eval_img if
                     eval_obj["class"].lower() in CLASSES_TO_COMPARE]

        # if there are detections in both images
        object_index = None
        if (not use_gt or len(ref_ious) > 0) and len(eval_ious) > 0:
            # get element with highest iou
            if use_gt:
                object_index = ref_ious.index(max(ref_ious))
                ref_inference_elem = ref_img[object_index]
            else:
                ref_inference_elem = object
            confidence_ref = ref_inference_elem["confidence"]

            # get element with highest iou
            eval_inference_elem = eval_img[eval_ious.index(max(eval_ious))]
            confidence_eval = eval_inference_elem["confidence"]

            # case when at least one detection of evaluation and reference image can be associated with the certain ground truth object
            if max(eval_ious) > iou_threshold and (not use_gt or max(ref_ious) > iou_threshold):
                # matched objects found
                # confidence loss is defined as the difference of the confidence scores of manipulated (evaluation) and original (reference) detections
                conf_loss = confidence_eval - confidence_ref
                # if confidence gains (confidence score of evaluation object is higher than the score of the reference object) are not allowed the confidence loss is set to 0
                if ignore_conf_gain:
                    if confidence_eval >= confidence_ref:
                        conf_loss = 0

            # case when only a detection in the evaluation image can be associated with the certain ground truth object
            elif use_gt and max(eval_ious) > iou_threshold >= max(ref_ious):
                confidence_ref = None
                if not ignore_conf_gain:
                    conf_loss = confidence_eval
                else:
                    conf_loss = 0

            # case when only a detection in the reference image can be associated with the certain ground truth object
            elif (not use_gt or max(ref_ious) > iou_threshold) and iou_threshold >= max(eval_ious):
                conf_loss = -1 * confidence_ref
                confidence_eval = None

            # case when no detections can be associated with the certain ground truth object
            else:
                conf_loss = 0
                confidence_eval = None
                confidence_ref = None

            conf_loss_list_img.append(
                {"ref_box": object.get("box"), "class": object.get("class"), "origin_frame": object.get("origin_frame"), "origin_scene": object.get("origin_scene"),
                 "gan_idx": object.get("gan_idx"), "color": object.get("foreground_object_color"),
                 "conf_loss_value": conf_loss, "confidence_eval": confidence_eval, "confidence_ref": confidence_ref})

        else:
            # case when only a detection in the evaluation image was found
            if use_gt and len(eval_ious) > 0 and len(ref_ious) == 0:
                # get element with highest iou
                eval_inference_elem = eval_img[eval_ious.index(max(eval_ious))]
                confidence_eval = eval_inference_elem["confidence"]
                confidence_ref = None

                if not ignore_conf_gain:
                    conf_loss = confidence_eval
                else:
                    conf_loss = 0

            # case when only a detection in the reference image was found
            elif (not use_gt or len(ref_ious) > 0) and len(eval_ious) == 0 and not ignore_no_detection:
                # get element with highest iou
                ref_inference_elem = ref_img[ref_ious.index(max(ref_ious))]
                confidence_ref = ref_inference_elem["confidence"]
                conf_loss = -1 * confidence_ref
                confidence_eval = None

            # case: no detection
            else:
                conf_loss = 0
                confidence_eval = None
                confidence_ref = None

            conf_loss_list_img.append(
                {"ref_box": object.get("box"), "class": object.get("class"), "origin_frame": object.get("origin_frame"), "origin_scene": object.get("origin_scene"),
                 "gan_idx": object.get("gan_idx"), "color": object.get("foreground_object_color"),
                 "conf_loss_value": conf_loss, "confidence_eval": confidence_eval, "confidence_ref": confidence_ref})

    return conf_loss_list_img


def output_mean(conf_loss_dict_list: list):
    """
            function calculates the mean ACL and its std

            INPUT:
                conf_loss_dict_list: list of ACLs
                result_path: path to store results

            OUTPUT:
                result_dict_list: dictionary that contains the input list and the calculated mean and std

    """

    conf_loss_list = []

    for conf_loss_dict in conf_loss_dict_list.values():
        for object_loss in conf_loss_dict:
            conf_loss_list.append(object_loss["conf_loss_value"])

    mean_conf_loss = None
    std_conf_loss = None
    ci_conf_loss = None

    if len(conf_loss_list) > 0:
        mean_conf_loss = np.nanmean(conf_loss_list)
        std_conf_loss = np.nanstd(conf_loss_list)
        ci_conf_loss = st.norm.interval(alpha=0.9999, loc=np.mean(conf_loss_list), scale=st.sem(conf_loss_list))

    print("mean = ", mean_conf_loss)
    print("std_dvt = ", std_conf_loss)
    print("ci = ", ci_conf_loss)

    result_dict_list = {
        "conf_loss_list_dict": conf_loss_dict_list,
        "mean": mean_conf_loss,
        "std_dvt": std_conf_loss,
        "ci": ci_conf_loss
    }

    return result_dict_list

def save_results_as_csv(conf_loss_filter_dict_result, reference_dataset, model_name, classes, final_directory):
    kpi_converter = ACLConverter(final_directory, reference_dataset)
    kpi_converter.export_kpi_to_csv("ACL", conf_loss_filter_dict_result, model_name, classes)
    kpi_converter.export_kpi_to_csv("Confidence", None, model_name, classes)

def get_result_summary(conf_loss_filter_dict_result):
    """
        function to get a summary of the ACL results

        INPUT:
            conf_loss_filter_dict_result: dictionary of detailed ACL results

        OUTPUT:
            result_summary: dictionary that contains the summary of ACL results sorted by the highest ACL (filter_name -> filter_values -> mean_acl)
    """
    result_summary = {}
    unsorted_list_max_acl = []
    for filter_name, filter_values in conf_loss_filter_dict_result.items():
        try:
            max_mean = max(filter_value["mean"] for filter_value in filter_values.values() if filter_value["mean"])
        except:
            max_mean = 0
        unsorted_list_max_acl.append((filter_name, max_mean))

    sorted_list_max_acl = sorted(unsorted_list_max_acl, key=lambda tup: tup[1], reverse = False)

    for filter_name_and_max_acl in sorted_list_max_acl:
        result_summary[filter_name_and_max_acl[0]] = {}
        for filter_value, results in conf_loss_filter_dict_result[filter_name_and_max_acl[0]].items():
            result_summary[filter_name_and_max_acl[0]][filter_value] = results["mean"]

    return result_summary

def collect_results(path_to_results):
    conf_loss_filter_dict = {}

    directory = path_to_results
    dir_to_sub_result = directory
    i = 0
    while os.path.exists(os.path.join(dir_to_sub_result, 'result_detailed.json')):

        print("reading: ", os.path.join(dir_to_sub_result, 'result_detailed.json'))
        with open(os.path.join(dir_to_sub_result, 'result_detailed.json')) as f:
            sub_results = json.load(f)

            for filter_name, filter_values in sub_results.items():
                if filter_name not in conf_loss_filter_dict:
                    conf_loss_filter_dict[filter_name] = {}
                for filter_value, results in filter_values.items():
                    if filter_value not in conf_loss_filter_dict[filter_name]:
                        conf_loss_filter_dict[filter_name][filter_value] = {}

                    for result in results:
                        image_name = result["image_name"]
                        compared_objects = result["compared_objects"]
                        conf_loss_filter_dict[filter_name][filter_value][image_name] = compared_objects

        print("deleting: ", dir_to_sub_result)
        shutil.rmtree(dir_to_sub_result)

        dir_to_sub_result = directory + f"_{i}"
        i += 1

    return conf_loss_filter_dict

class DatasetException(Exception):
    """
        custom exception class for all problems arising cause of the datasets or defined filters
    """
    pass
