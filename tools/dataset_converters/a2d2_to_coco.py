import json
import argparse
import os
from pathlib import Path

import mmcv
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 annotations to mmdetection format')
    parser.add_argument('a2d2_path', help='A2D2 dataset path')
    parser.add_argument('-ld', '--label-dir-name', help='A2D2 dataset path', type=str, default='camera_frontcenter')
    parser.add_argument('-id', '--image-dir-name', help='A2D2 dataset path', type=str, default='image_undistort')
    parser.add_argument('-c', '--classes', help='A2D2 classes file name', type=str, default='class_list.json')
    parser.add_argument('-u', '--undistort-images', action='store_true')
    parser.add_argument('-dn', '--dir-names', help='Names of A2D2 folders to convert', nargs='+', type=str)
    parser.add_argument('-od', '--out-dir', help='output path')
    parser.add_argument('-of', '--out-file', help='output file name')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args

def collect_image_infos(path, dir_names=None):
    img_infos = []

    images_generator = mmcv.scandir(path, recursive=True)
    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if (dir_names is None or (
                dir_names is not None
                and image_path.lower().startswith(dir_names))) and image_path.lower().endswith('png') and \
                len(image_path.split('/')) > 1 and image_path.split('/')[1] == 'image_undistort':
            image_path = os.path.join(path, image_path)
            img_pillow = Image.open(image_path)

            image_path_split = image_path.split('_')
            image_id = int(image_path_split[len(image_path_split) - 1].split('.')[0])

            img_info = {
                'file_name': image_path,
                'width': img_pillow.width,
                'height': img_pillow.height,
                'id': image_id
            }

            img_infos.append(img_info)


    return img_infos


def collect_image_annos(path, categories, dir_names=None, label_dir_name='label3D'):
    img_annos = []

    images_generator = mmcv.scandir(path, recursive=True)

    annotation_id = 0 # TODO: Was ist das genau für ne id???

    for image_path in mmcv.track_iter_progress(list(images_generator)):
        if (dir_names is None or (
                dir_names is not None
                and image_path.lower().startswith(dir_names))) and image_path.lower().endswith('json') and \
                len(image_path.split('/')) > 1 and image_path.split('/')[1] == label_dir_name:

            anno_path = os.path.join(path, image_path)
            image_path_split = image_path.split('_')
            image_id = int(image_path_split[len(image_path_split) - 1].split('.')[0])

            with open(anno_path) as f:
                anno_data = json.load(f)

                for box_id, a2d2_anno in anno_data.items():

                    coco_anno = cvt_a2d2_to_coco_anno(a2d2_anno, categories)

                    img_anno = {
                        'segmentation': [], #TODO
                        'area': coco_anno['area'],
                        'iscrowd': coco_anno['iscrowd'],
                        'category_id': coco_anno['category_id'],
                        'bbox': coco_anno['bbox'],
                        'image_id': image_id,
                        'id': annotation_id
                    }

                    img_annos.append(img_anno)
                    annotation_id += 1

    return img_annos

def cvt_a2d2_to_coco_anno(a2d2_anno, categories):

    category_id = None
    for category in categories:
        if category['name'].lower() == 'car' and a2d2_anno['class'] == 'VanSUV':
            category_id = category['id']
        if category['name'].lower() == a2d2_anno['class'].lower():
            category_id = category['id']

    coco_annotations = {
            "category_id": category_id,
            "bbox": [a2d2_anno["2d_bbox"][0], a2d2_anno["2d_bbox"][1],
                     a2d2_anno["2d_bbox"][2] - a2d2_anno["2d_bbox"][0],
                     a2d2_anno["2d_bbox"][3] - a2d2_anno["2d_bbox"][1]],  # [x, y, width, height]
            "area": (a2d2_anno["2d_bbox"][2] - a2d2_anno["2d_bbox"][0]) * (a2d2_anno["2d_bbox"][3] - a2d2_anno["2d_bbox"][1]),
            "iscrowd": 0 #TODO
        }

    return coco_annotations

def cvt_categories(classes_path):
    categories = []
    with open(classes_path) as f:
        classes = json.load(f)
        id = 0
        appended_categories = []

        for category in classes.values():
            try:
                cs = category.split(' ')
                int(cs[len(cs) - 1])
                category_name = ' '.join(cs[0:(len(cs) - 1)]).lower()
            except:
                category_name = category.lower()

            if category_name not in appended_categories:
                categories.append({
                    'supercategory': category_name,
                    'id': id,
                    'name': category_name
                })
                appended_categories.append(category_name)
                id += 1

    return categories

def main():
    args = parse_args()

    # 1 load categories
    categories = cvt_categories(os.path.join(args.a2d2_path, args.classes))

    # 2 load image list info and annotation info
    img_infos = collect_image_infos(args.a2d2_path, args.dir_names)
    img_annos = collect_image_annos(args.a2d2_path, categories, args.dir_names, args.label_dir_name)

    # 3 dump
    coco_info = dict()
    coco_info['images'] = img_infos
    coco_info['annotations'] = img_annos
    coco_info['categories'] = categories

    if args.out_dir is None:
        base = Path(__file__).parent.parent.parent
        out_dir = os.path.join(base, 'data/a2d2')
    else:
        out_dir = args.out_dir

    save_dir = os.path.join(out_dir, 'annotations')
    mmcv.mkdir_or_exist(save_dir)
    save_path = os.path.join(save_dir, args.out_file)
    mmcv.dump(coco_info, save_path)
    print(f'save json file: {save_path}')

    # 4 store images
    # TODO: Undistort
    #  -> Vorsicht beim erneuten abspeichern, Pfade der Bilder sind bereits in coco_info['images'] hinterlegt und sollten angepasst werden
    #  -> Kann aber evtl. vernachlässigt werden


if __name__ == '__main__':
    main()