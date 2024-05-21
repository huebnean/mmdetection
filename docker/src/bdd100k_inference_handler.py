from flask import Flask, request, jsonify
from utils.bucket_operations import BucketManager
from utils import utils
from inference import inference
import os
import sys
import shutil
import json
from pathlib import Path
from tools.dataset_converters import bdd100k_to_coco
from dataset import ImageDetectionDataset
from kpi.kpi_calculator import KpiCalculator
class BDD100KInference:
    def __init__(self) -> None:
        """
        Initializing the bucket, flask app and the paths
        """
        self.app = Flask(__name__)
        self.bucket_name = 'bertrandt-input'
        self.bucket_manager = BucketManager(self.bucket_name, self.app)
        self.bdd100k_categories_json_path = "docker/src/bdd_10k_categories.json"


    def create_download_folder_tree(self):
        """
        Create temporary folder to download the data and the results
        """
        self.downloads_path = "downloads"
        utils.create_folder(self.downloads_path)
        self.labels_download_path = os.path.join(self.downloads_path, 'labels')
        utils.create_folder(self.labels_download_path)
        self.images_download_path = os.path.join(self.downloads_path, 'val/images')
        utils.create_folder(self.images_download_path)
        self.results_path = os.path.join(self.downloads_path, 'val/results')
        utils.create_folder(self.results_path)
        shutil.copy(self.bdd100k_categories_json_path, self.downloads_path)

    def create_dataset_dict(self):
        self.dataset_dict = {}
        for image_path in Path(self.images_download_path).glob('*.jpg'):
            self.dataset_dict[image_path.stem] = str(image_path)


    def download_dataset_and_labels_from_bucket(self):
        """
        Read and download the bdd100k dataset from the s3 bucket.

        Raises:
            Exception: Raise an exception if the bucket is empty.
        """
        try:
            paginator = self.bucket_manager.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=f'datasets/bdd100k_val/bdd100k/bdd100k/images/100k/val/')
            label_page = paginator.paginate(Bucket=self.bucket_name, Prefix='datasets/bdd100k_val/bdd100k_labels_images_val.json')

            for objects in label_page:
                if objects['KeyCount'] == 0:
                    raise Exception("No Labels Found")
                else:
                    labels_path = objects['Contents'][0]['Key']
                    print(labels_path)
            self.bucket_manager.download_folder_from_s3(labels_path, self.labels_download_path)
            print(labels_path)

            for objects in pages:
                images=[]
                if objects['KeyCount'] == 0:
                    print("The bucket is empty.")
                else:
                    print("Listing objects in bucket:")
                    print(len(objects['Contents']))
                    ind = 0
                    for obj in objects['Contents']:
                        print(f"Key: {obj['Key']}, Last Modified: {obj['LastModified']}, Size: {obj['Size']}")
                        if ind == 1:
                            images.append(obj['Key'])
                            self.bucket_manager.download_folder_from_s3(obj['Key'], self.images_download_path)
                        ind += 1
        except self.bucket_manager.s3.exceptions.NoSuchBucket:
            print("Bucket does not exist.")
        except Exception as e:
            print(f"Error: {e}")

    def convert_bdd100k_labels_to_coco(self):
        """
        Convert the bdd100k labels to coco format before inferencing
        """
        sys.argv[1:] = [self.downloads_path,"-lp",  "labels/bdd100k_labels_images_val.json", "-ip", "val/images",
                        "-c", "bdd_10k_categories.json", "-od", self.results_path, "-of", "converted_labels.json"]
        bdd100k_to_coco.main()

    def perform_inference(self):
        """
        Perform the inference of the dataset and upload the result to the bucket
        """
        self.inference_result = inference(self.downloads_path)
        # self.bucket_manager.upload_folder_to_s3(self.results_path, 'datasets/bdd100k_val/')

    def kpi_calculation(self):
        """_Generates the Dataset for ground truth and inference and calculates KPI's for it
        """
        ground_truth_dict = ImageDetectionDataset.get_coco_dataset(self.results_path + "/annotations/converted_labels.json")
        self.create_dataset_dict()
        inference_dict = ImageDetectionDataset(DatasetDict=self.dataset_dict, GroundTruth=self.inference_result)

        KpiCalculator().precision_loss_calculation(ground_truth_dict, inference_dict)

    def flush_downloaded_data(self):
        """Removes the downloaded data
        """
        shutil.rmtree(self.downloads_path)

    def run(self):
        """
        Run all the function accordingly
        """
        self.create_download_folder_tree()
        self.download_dataset_and_labels_from_bucket()
        self.convert_bdd100k_labels_to_coco()
        self.perform_inference()
        self.kpi_calculation()

if __name__ == '__main__':
    bdd100k_inference = BDD100KInference()
    bdd100k_inference.run()


