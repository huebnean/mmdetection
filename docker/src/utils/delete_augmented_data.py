import os

'''  
    Simple script to delete some data.
    Be careful when using this because all files will be permanently deleted!!!
    
    Parameter:
        FOLDER_PATH: Folder that contains all datasets which has to be deleted
        DATASET_NAMES: All names of datasets which has to be deleted
        EXCLUDED_FILE_NAMES: All file names that should not be deleted in each dataset (e.g. GroundTruth.json)
        REMEMBER_IMAGES: Integer for a number of images that should not be deleted too
'''

FOLDER_PATH = r"/home/hubnera/PycharmProjects/ml_testingtool/data_preparation/datasets/manipulated_datasets"
DATASET_NAMES = []
EXCLUDED_FILE_NAMES = ["DatasetDict.json", "DatasetInfo.json", "GroundTruth.json", "semantic_segmentation.json"]
REMEMBER_IMAGES = 1

def delete_data(dataset_names: str = "", excluded_file_names: list = EXCLUDED_FILE_NAMES, remembered_images: int = REMEMBER_IMAGES):
    dataset_names_list = dataset_names.split(",")
    for dataset_name in dataset_names_list:
        dataset_path = os.path.join(str(FOLDER_PATH), str(dataset_name).strip())

        i = 0
        for file in os.listdir(dataset_path):

            file_path = os.path.join(dataset_path, file)
            if i < remembered_images and file.endswith(".png"):
                i += 1
                print(f"UNTOUCHED: {file_path}")
                continue

            if os.path.isfile(file_path) and file not in excluded_file_names:
                os.remove(file_path)
                print(f"DELETED: {file_path}")
            else:
                print(f"UNTOUCHED: {file_path}")
