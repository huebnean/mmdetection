import gzip
import pandaset
import json
import argparse
import os
from pathlib import Path

import cv2
import mmcv
from PIL import Image

CAMERA_ID_TO_CAMERA_DIR = {
    0: 'front_left_camera',
    1: 'front_camera',
    2: 'front_right_camera',
    3: 'right_camera',
    4: 'back_camera',
    5: 'left_camera'
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Pandaset annotations to mmdetection format')
    parser.add_argument('pandaset_path', help='Pandaset dataset path')
    parser.add_argument('-cd', '--camera-dir-name', help='Pandaset camera path', type=str, default='camera')
    parser.add_argument('-ld', '--label-dir-name', help='Pandaset annotation path', type=str, default='annotations')
    parser.add_argument('-c', '--classes', help='Pandaset classes file name', type=str, default=os.path.join('semseg', 'classes.json'))
    parser.add_argument('-dnc', '--dir-names-camera', help='Names of Pandaset camera folders to convert', nargs='+', type=str)
    parser.add_argument('-dn', '--dir-names', help='Names (Numbers) of Pandaset folders to convert', nargs='+', type=str)
    parser.add_argument('-u', '--undistort-images', action='store_true')
    parser.add_argument('-od', '--out-dir', help='output path', default="data/pandaset/")
    parser.add_argument('-of', '--out-file', help='output file name', default="pandaset_in_coco.json")
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args

def collect_image_infos(pandaset_path, camera_dir_name, dir_names, dir_names_camera):
    img_infos = []
    img_id_dict = {}
    image_id = 1

    pandaset_items = os.listdir(pandaset_path)
    pandaset_directories = [item for item in pandaset_items if os.path.isdir(os.path.join(pandaset_path, item))]

    for pd_dir_name in pandaset_directories:
        if dir_names is None or pd_dir_name in dir_names:
            images_path = os.path.join(pandaset_path, pd_dir_name, camera_dir_name)

            camera_items = os.listdir(images_path)
            camera_directories = [item for item in camera_items if os.path.isdir(os.path.join(images_path, item))]

            for c_dir_name in camera_directories:
                if dir_names_camera is None or c_dir_name in dir_names_camera:
                    camera_path = os.path.join(images_path, c_dir_name)

                    img_id_dict[c_dir_name] = {}

                    # Loop over all files in the directory
                    for image_file_name in os.listdir(camera_path):
                        # Get the full file path
                        image_file_path = os.path.join(camera_path, image_file_name)
                        if image_file_name.split('.')[1] != "jpg":
                            continue
                        img_id_dict[c_dir_name][int(image_file_name.split('.')[0])] = image_id

                        # Check if the file is a regular file (i.e., not a directory)
                        if os.path.isfile(image_file_path) and image_file_name.endswith('.jpg'):

                            img_pillow = Image.open(image_file_path)

                            img_info = {
                                'file_name': image_file_path,
                                'width': img_pillow.width,
                                'height': img_pillow.height,
                                'id': image_id
                            }

                            img_infos.append(img_info)

                            image_id += 1

    return img_infos, img_id_dict


def convert_to_2d_bb(points_3d, h, w):
    x_min = w
    x_max = 0
    y_min = h
    y_max = 0

    for i in range(len(points_3d)):

        if points_3d[i][0] < x_min:
            if points_3d[i][0] < 0:
                x_min = 0
            else:
                x_min = points_3d[i][0]

        if points_3d[i][0] > x_max:
            if points_3d[i][0] > w:
                x_max = w
            else:
                x_max = points_3d[i][0]

        if points_3d[i][1] < y_min:
            if points_3d[i][1] < 0:
                y_min = 0
            else:
                y_min = points_3d[i][1]

        if points_3d[i][1] > y_max:
            if points_3d[i][1] > h:
                y_max = h
            else:
                y_max = points_3d[i][1]

    return int(x_min), int(y_min), int(x_max), int(y_max)

def collect_image_annos(pandaset_path, categories, image_id_dict):
    img_annos = []
    annotation_id = 0

    fails = 0

    dataset = pandaset.DataSet(pandaset_path)
    print((dataset.sequences()))

    for camera_name in CAMERA_ID_TO_CAMERA_DIR.values():
        for data_idx in (dataset.sequences()):
            dataset_idx = data_idx
            print("Start loading data - idx: ", dataset_idx)
            seq = dataset[dataset_idx]

            try:
                seq.load()
                cuboids = seq.cuboids

                if cuboids.data == []:
                    fails += 1
                    print("No cuboids - idx: ", dataset_idx)
                    continue


            except Exception as e:
                fails += 1
                print("Error: No cuboids - idx: ", dataset_idx)
                continue

            print("Start converting - idx: ", dataset_idx)
            selected_camera = seq.camera[camera_name]
            for time_idx in range(len(seq.timestamps.data)):
                pil_image = seq.camera[camera_name][time_idx]

                w, h = pil_image.size
                cub = cuboids[time_idx]
                for idx, row in cub.iterrows():
                    if (cub["camera_used"][idx] == 0 or cub["camera_used"][idx] == 1 or cub["camera_used"][idx] == 2 or
                        cub["camera_used"][idx] == -1) and (
                            cub["label"][idx] == "Car" or cub["label"][idx] == "Pedestrian" or cub["label"][
                        idx] == "Pedestrian with Object") and (
                            cub["cuboids.sensor_id"][idx] == -1 or cub["cuboids.sensor_id"][idx] == 1):
                        box = pandaset.geometry.center_box_to_corners([row['position.x'],
                                                                              row['position.y'],
                                                                              row['position.z'],
                                                                              row['dimensions.x'],
                                                                              row['dimensions.y'],
                                                                              row['dimensions.z'],
                                                                              row['yaw']])
                        projected_points2d, camera_points_3d, inner_indices = pandaset.geometry.projection(
                            lidar_points=box,
                            camera_data=selected_camera[time_idx],
                            camera_pose=selected_camera.poses[time_idx],
                            camera_intrinsics=selected_camera.intrinsics,
                            filter_outliers=False)

                        if len([x for x in camera_points_3d[2] if x < 0]) > 0:
                            continue

                        bb_x1, bb_y1, bb_x2, bb_y2 = convert_to_2d_bb(projected_points2d, h, w)
                        #print(projected_points2d)
                        #print([bb_x1, bb_y1, bb_x2, bb_y2])

                        if bb_x2 == 0 and bb_y2 == 0 and bb_x1 == w and bb_y1 == h or bb_x1 == 0 and bb_x2 == 0\
                                or bb_y1 == 0 and bb_y2 == 0 or bb_x1 == w and bb_x2 == w\
                                or bb_y1 == h and bb_y2 == h:
                            #print("Ignore bb")
                            continue

                        #cv2.rectangle(img, (bb_x1, bb_y1), (bb_x2, bb_y2), (0, 255, 0), 2)
                        #print(f"{camera_name}: print {idx}. box on {time_idx}. img")

                        category_id = None
                        for category in categories:
                            if category['name'].lower() == cub["label"][idx].lower():
                                category_id = category['id']

                        img_anno = {
                            'segmentation': [],  # TODO
                            'area': ((bb_x2 - bb_x1) * (bb_y2 - bb_y1)),
                            'iscrowd': 0,  # TODO
                            'category_id': category_id,
                            'bbox': [bb_x1, bb_y1, bb_x2 - bb_x1, bb_y2 - bb_y1],
                            'image_id': image_id_dict[camera_name][time_idx],
                            'id': annotation_id
                        }
                        annotation_id += 1

                        img_annos.append(img_anno)

            dataset.unload(dataset_idx)
            print("Sequence unloaded - ", dataset_idx)

                #cv2.imshow('image', img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

    print("FAILS: ", fails)
    return img_annos

def cvt_categories(pandaset_path, label_dir_name, classes):
    categories = []

    # Get a list of all items in the directory
    items = os.listdir(pandaset_path)

    # Filter the items to only include directories
    directories = [item for item in items if os.path.isdir(os.path.join(pandaset_path, item))]

    # Get the first directory in the list
    if directories:
        first_directory = directories[0]
    else:
        print(pandaset_path, 'does not contain any directories')
        return None

    with open(os.path.join(pandaset_path, first_directory, label_dir_name, classes)) as f:
        classes = json.load(f)

        for category_id, category_name in classes.items():
            category_name = category_name.lower()
            categories.append({
                'supercategory': category_name,
                'id': category_id,
                'name': category_name
            })

    return categories

def main():
    args = parse_args()

    # 1 load categories
    categories = cvt_categories(args.pandaset_path, args.label_dir_name, args.classes)

    # 2 load image list info and annotation info
    img_infos, img_id_dict = collect_image_infos(args.pandaset_path, args.camera_dir_name, args.dir_names, args.dir_names_camera)
    img_annos = collect_image_annos(args.pandaset_path, categories, img_id_dict)

    # 3 dump
    coco_info = dict()
    coco_info['images'] = img_infos
    coco_info['annotations'] = img_annos
    coco_info['categories'] = categories

    if args.out_dir is None:
        base = Path(__file__).parent.parent.parent
        out_dir = os.path.join(base, 'data', 'pandaset')
    else:
        out_dir = args.out_dir

    save_dir = os.path.join(out_dir, 'annotations')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, args.out_file)
    with open(save_path, 'w') as f: json.dump(coco_info, f)
    print(f'save json file: {save_path}')

    # 4 store images
    # TODO: Undistort
    #  -> Vorsicht beim erneuten abspeichern, Pfade der Bilder sind bereits in coco_info['images'] hinterlegt und sollten angepasst werden
    #  -> Kann aber evtl. vernachlässigt werden


if __name__ == '__main__':
    main()