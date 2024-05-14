import boto3
from inference import inference

bucket_name = 'bertrandt-input'

s3 = boto3.client('s3', verify=False)
try:
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='datasets/bdd100k_val/bdd100k/bdd100k/images/100k/val/')
    for objects in pages:
        images=[]
        if objects['KeyCount'] == 0:
            print("The bucket is empty.")
        else:
            print("Listing objects in bucket:")
            print(len(objects['Contents']))
            for obj in objects['Contents']:
                print(f"Key: {obj['Key']}, Last Modified: {obj['LastModified']}, Size: {obj['Size']}")
                images.append(obj['Key'])
            inference(images)

except s3.exceptions.NoSuchBucket:
    print("Bucket does not exist.")
except Exception as e:
    print(f"Error: {e}")