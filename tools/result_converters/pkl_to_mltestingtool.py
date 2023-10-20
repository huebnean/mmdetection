import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert A2D2 annotations to mmdetection format')
    parser.add_argument('a2d2_path', help='A2D2 dataset path')
    parser.add_argument('-l', '--label-dir-name', help='A2D2 dataset path', type=str, default='label3D')
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

with open(r'C:\Users\huebnean\PycharmProjects\mmdetection\results\res.pkl', 'rb') as f:
    data = pickle.load(f)
    data = data

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
        out_dir = os.path.join(base, 'data\\a2d2')
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