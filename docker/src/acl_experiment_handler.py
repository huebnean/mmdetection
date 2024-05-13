import boto3

bucket_name = 'bertrandt-input'

s3 = boto3.client('s3')
try:
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix='datasets/bdd100k_val/bdd100k/bdd100k/images/100k/val/')
    for objects in pages:
        if objects['KeyCount'] == 0:
            print("The bucket is empty.")
        else:
            print("Listing objects in bucket:")
            print(len(objects['Contents']))
            for obj in objects['Contents']:
                print(f"Key: {obj['Key']}, Last Modified: {obj['LastModified']}, Size: {obj['Size']}")
except s3.exceptions.NoSuchBucket:
    print("Bucket does not exist.")
except Exception as e:
    print(f"Error: {e}")