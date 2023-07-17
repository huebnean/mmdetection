import json
import os

CLASSES_TO_CONVERT = {0: 0, 2: 1, 3: 1, 4: 1, 6: 2, 9: 3}

def convert(source, dest):

    # Opening JSON file
    f = open(source)
    data = json.load(f)

    train_txt = open(f"{dest}/train.txt", "w")

    img_w = data["images"][0]["width"]
    img_h = data["images"][0]["height"]

    annotations = dict()
    for annotation in data["annotations"]:
        if annotation["image_id"] not in annotations:
            annotations[annotation["image_id"]] = list()

        print(annotation["category_id"])
        if annotation["category_id"] not in CLASSES_TO_CONVERT.keys():
            continue

        x = (annotation["bbox"][0] + (annotation["bbox"][2] / 2)) / img_w
        y = (annotation["bbox"][1] + (annotation["bbox"][3] / 2)) / img_h
        w = annotation["bbox"][2] / img_w
        h = annotation["bbox"][3] / img_h

        annotations[annotation["image_id"]].append(f"{CLASSES_TO_CONVERT[annotation['category_id']]} {x} {y} {w} {h}\n")

    for image in data['images']:
        print(image["file_name"])
        dir, filename = os.path.split(image["file_name"])
        img_name = filename.split(".")[0]

        train_txt.write(image["file_name"] + "\n")

        anno_txt = open(f"{dest}/data/{img_name}.txt", "w")
        for annotation in annotations[image["id"]]:
            anno_txt.write(annotation)
        anno_txt.close()

    train_txt.close()
    f.close()

convert(r"/home/hubnera/PycharmProjects/mmdetection/data/bdd100k/annotations/bdd100k_coco_train.json", r"/home/hubnera/PycharmProjects/mmdetection/data/darknet_train")