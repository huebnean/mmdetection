import os
import boto3
import logging

# Initialize the S3 client
s3 = boto3.client('s3')
logging.basicConfig(level=logging.INFO)

class BucketManager():
    def __init__(self, bucket_name, app):
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3', verify=False)
        self.app = app

    def check_folder_recursive(self, folder_path):
        """
        Recursively create a folder if it doesn't exist.

        Parameters:
            folder_path (str): The path of the folder to create.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")

    def download_folder_from_s3(self, folder_key, local_dir):

        self.check_folder_recursive(local_dir)

        try:
            # List all objects within the folder
            self.app.logger.info(f"Trying to download {folder_key} from bucket {self.bucket_name}.")

            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=folder_key)

            self.app.logger.info(f"Bucket content:")
            for obj in response.get('Contents', []):
                self.app.logger.info(f"Key: {obj['Key']}, Last Modified: {obj['LastModified']}, Size: {obj['Size']}")

            # Iterate through each object
            for obj in response.get('Contents', []):
                # Get the object key (path)
                file_key = obj['Key']

                # Generate local file path
                local_file_path = os.path.join(local_dir, os.path.basename(file_key))

                # Download the object
                self.s3.download_file(self.bucket_name, file_key, local_file_path)

                self.app.logger.info(f"Downloaded {file_key} to {local_file_path}")

            self.app.logger.info("Folder download completed.")
        except Exception as e:
            # Handle exceptions
            self.app.logger.info(f"Error downloading folder from S3: {e}")


    def get_file_content_from_s3(self, file_key):
        try:
            # Get the file object
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)

            # Read the contents of the file
            file_content = response['Body'].read()

            # Return the file content
            return file_content
        except Exception as e:
            # Handle exceptions
            print(f"Error getting file from S3: {e}")
            return None

    def upload_folder_to_s3(self, local_folder, s3_folder):
        """
        Uploads all files in a local folder to a specified folder in an S3 bucket.

        Args:
        local_folder (str): Path to the local folder to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): Path within the bucket to upload the files to.
        """
        for root, dirs, files in os.walk(local_folder):
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_folder)
                s3_path = os.path.join(s3_folder, relative_path).replace("\\", "/")  # Use "/" for S3 paths

                # Upload the file to S3
                self.s3.upload_file(local_path, self.bucket_name, s3_path)
                self.app.logger.info(f'Uploaded {local_path} to s3://{self.bucket_name}/{s3_path}')
