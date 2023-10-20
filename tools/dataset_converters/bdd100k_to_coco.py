'''

This script converts bdd100k data to coco format.

Example: python bdd100k_to_coco.py data/bdd100k/ -lp bdd100k_labels/labels/bdd100k_labels_images_val.json -ip bdd100k_images/images/100k/val -od data/bdd100k/coco

'''


import json
import argparse
import os
from pathlib import Path

import mmcv
import cv2
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BDD100k annotations to mmdetection format')
    parser.add_argument('bdd100k_path', help='BDD100k dataset path')
    parser.add_argument('-lp', '--label-path', help='BDD100k label path', type=str, default='')
    parser.add_argument('-ip', '--image-dir', help='BDD100k image directory', type=str, default='')
    parser.add_argument('-c', '--classes', help='BDD100k classes file name', type=str, default='categories.json')
    parser.add_argument('-u', '--undistort-images', action='store_true')
    parser.add_argument('-od', '--out-dir', help='output path')
    parser.add_argument('-of', '--out-file', help='output file name', default='data.json')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def cvt_bdd100k_to_coco_anno(bdd100k_label, categories):

    category_id = None
    for category in categories:
        if category['name'].lower() == bdd100k_label['category'].lower():
            category_id = category['id']

    coco_annotations = {
            "category_id": category_id,
            "bbox": [bdd100k_label["box2d"]['x1'], bdd100k_label["box2d"]['y1'],
                     bdd100k_label["box2d"]['x2'] - bdd100k_label["box2d"]['x1'],
                     bdd100k_label["box2d"]['y2'] - bdd100k_label["box2d"]['y1']],  # [x, y, width, height]
            "area": (bdd100k_label["box2d"]['x2'] - bdd100k_label["box2d"]['x1']) * (bdd100k_label["box2d"]['y2'] - bdd100k_label["box2d"]['y1']),
            "iscrowd": 0, #TODO
            "occluded": int(bdd100k_label['attributes']['occluded']),
            "truncated": int(bdd100k_label['attributes']['truncated']),
            "trafficLightColor": bdd100k_label['attributes']['trafficLightColor'],
            "id": int(bdd100k_label['id'])
        }

    return coco_annotations

def collect_image_infos_and_annos(base_path, label_path, image_dir, categories):
    img_infos = []
    img_annos = []

    image_id = 0

    with open(os.path.join(base_path, label_path)) as f:
        annos = json.load(f)

    for anno_data in annos:
        image_path = os.path.join(base_path, image_dir, anno_data['name'])

        try:
            image = cv2.imread(image_path)
        except:
            print("Image with the following path does not exist: ", image_path)
            continue

        if os.path.exists(image_path) and image is not None:
            img_pillow = Image.open(image_path)

            img_info = {
                'file_name': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
                'id': image_id
            }

            img_infos.append(img_info)

            if 'labels' in anno_data:
                img_annos += collect_image_annos(anno_data['labels'], image_id, categories)

            image_id += 1

            print("Image with the following path appended: ", image_path)

        else:
            print("Image with the following path does not exist: ", image_path)

    print("Images in dataset: ", image_id)

    return img_infos, img_annos


def collect_image_annos(labels, image_id, categories):
    img_annos = []

    for bdd100k_label in labels:
        if 'box2d' not in bdd100k_label:
            continue
        # TODO: else (poly2d)
        coco_anno = cvt_bdd100k_to_coco_anno(bdd100k_label, categories)

        img_anno = {
            'segmentation': [], #TODO
            'area': coco_anno['area'],
            'iscrowd': coco_anno['iscrowd'],
            'category_id': coco_anno['category_id'],
            'bbox': coco_anno['bbox'],
            'image_id': image_id,
            'id': coco_anno['id']
        }

        img_annos.append(img_anno)

    return img_annos

def cvt_bdd100k_to_coco_anno(bdd100k_label, categories):

    category_id = None
    for category in categories:
        if category['name'].lower() == bdd100k_label['category'].lower():
            category_id = category['id']

    coco_annotations = {
            "category_id": category_id,
            "bbox": [bdd100k_label["box2d"]['x1'], bdd100k_label["box2d"]['y1'],
                     bdd100k_label["box2d"]['x2'] - bdd100k_label["box2d"]['x1'],
                     bdd100k_label["box2d"]['y2'] - bdd100k_label["box2d"]['y1']],  # [x, y, width, height]
            "area": (bdd100k_label["box2d"]['x2'] - bdd100k_label["box2d"]['x1']) * (bdd100k_label["box2d"]['y2'] - bdd100k_label["box2d"]['y1']),
            "iscrowd": 0, #TODO
            "occluded": int(bdd100k_label['attributes']['occluded']),
            "truncated": int(bdd100k_label['attributes']['truncated']),
            "trafficLightColor": bdd100k_label['attributes']['trafficLightColor'],
            "id": int(bdd100k_label['id'])
        }

    return coco_annotations

def load_categories(classes_path):
    with open(classes_path) as f:
        classes = json.load(f)
        categories = classes['categories']

    return categories

def main():
    #TODO: Progress bar!!!

    args = parse_args()

    # 1 load categories -> manuelles Erstellen der Datei erforderlich
    categories = load_categories(os.path.join(args.bdd100k_path, args.classes))

    # 2 load image list info and annotation info
    img_infos, img_annos = collect_image_infos_and_annos(args.bdd100k_path, args.label_path, args.image_dir, categories)

    # 3 dump
    coco_info = dict()
    coco_info['images'] = img_infos
    coco_info['annotations'] = img_annos
    coco_info['categories'] = categories

    if args.out_dir is None:
        base = Path(__file__).parent.parent.parent
        out_dir = os.path.join(base, 'data', 'bdd100k')
    else:
        out_dir = args.out_dir

    save_dir = os.path.join(out_dir, 'annotations')
    #mmcv.mkdir_or_exist(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, args.out_file)
    #json.dump(coco_info, save_path)
    with open(save_path, "w") as outfile:
        json.dump(coco_info, outfile)
    print(f'save json file: {save_path}')

    # 4 store images
    # TODO: Undistort
    #  -> Vorsicht beim erneuten abspeichern, Pfade der Bilder sind bereits in coco_info['images'] hinterlegt und sollten angepasst werden
    #  -> Kann aber evtl. vernachlässigt werden


if __name__ == '__main__':
    main()