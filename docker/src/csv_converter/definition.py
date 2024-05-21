'''
    Datasets
'''
CSV_DATASET_LABEL_FILE_NAME = "dataset_label.csv"
CSV_DATASET_BB_FILE_NAME = "dataset_bb_class.csv"
DATASET_NAME_TO_CONVERTER = {
    "a2d2_dataset": "A2D2CSVConverter",
    "bdd100k_dataset": "CocoCSVConverter",
    "rotate_0_50": "CocoCSVConverter",
    "rotate_0": "CocoCSVConverter",
    "synthetic_dataset_5000_rand": "A2D2CSVConverter"
}
DATASET_LABEL_FIRST_ROW = ["dataset_name", "image_name", "object_id", "label_name", "label_value"]
DATASET_BB_FIRST_ROW = ["dataset_name", "image_name", "object_id", "bounding_box", "class", "center_x", "center_y", "width", "height"]

DATASET_BB_NAME = {
    "a2d2_dataset": "box",
    "bdd100k_dataset": "box",
    "synthetic_dataset_5000_rand": "box",
    "rotate_0_50": "box",
    "rotate_0": "box"
}

DATASET_CLASS_NAME = {
    "a2d2_dataset": "Label",
    "bdd100k_dataset": "class",
    "synthetic_dataset_5000_rand": "Label",
    "rotate_0_50": "class",
    "rotate_0": "class"
}

LABEL_NAMES = {
    "a2d2_dataset": ["distance_bin", "orientation_bin"],
    "bdd100k_dataset": [],
    "rotate_0_50": ["area"],
    "rotate_0": ["area"],
    "synthetic_dataset_5000_rand": ["distance_bin", "orientation_bin"]
}

'''
    Coco
'''

'''
    A2d2
'''
A2D2_BINS = [
    (5.891, 0.393),  # ]337.5°, 22.5°]
    (0.393, 1.178),  # ]22.5°, 67.5°]
    (1.178, 1.964),  # ]67.5°, 112.5°]
    (1.964, 2.749),  # ]112.5°, 157.5°]
    (2.749, 3.534),  # ]157.5°, 202.5°]
    (3.534, 4.320),  # ]202.5°, 247.5°]
    (4.320, 5.105),  # ]247.5°, 292.5°]
    (5.105, 5.891)  # ]292.5°, 337.5°]
]
# list for distance of the object
A2D2_DISTANCE = [0, 10, 20, 30, 40, 50]


'''
    KPIs
'''
CSV_KPI_FILE_NAME_PREFIX = "KPI"
KPI_FIRST_ROW = ["dataset_name", "image_name", "object_id", "model_name", "metrik_name", "metrik_value"]