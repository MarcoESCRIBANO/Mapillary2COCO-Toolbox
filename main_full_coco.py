"""
    This file shows how to load and use the dataset
"""

import json
import os

import numpy as np
import warnings

from PIL import Image
import multiprocessing as mp
import pycococreatortools
import datetime

### Set parameters ###

# FULL_CPU: False will reduce the cpu count by CPU_REDUCTION
# FULL_CPU: True will use the maximal capabilities of your cpu (can slow down your computer) 
FULL_CPU = True

# Reduce the cpu count by 4 when FULL_CPU is False
CPU_REDUCTION = 4 

# For running large dataset (split the execution in batch to be able to run in part, to not restart from scratch if it crash)
BATCH_SIZE = 600 

# For resume after an interruption (num of the batch where it will resume) 
STARTING_BATCH = 0

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    }
]

CATEGORIES = [
    {
        "id": 0,
        "name": "person",
        "supercategory": "person",
        "color": [
                220,
                20,
                60
            ]
    },
    {
        "id": 1,
        "name": "bicycle",
        "supercategory": "vehicle",
        "color": [
                255,
                0,
                0
            ]
    },
    {
        "id": 2,
        "name": "car",
        "supercategory": "vehicle",
        "color": [
                0,
                0,
                142
            ]
    },
    {
        "id": 3,
        "name": "motorcycle",
        "supercategory": "vehicle",
        "color": [
                0,
                0,
                230
            ]
    },
    {
        "id": 4,
        "name": "airplane",
        "supercategory": "vehicle",
        "color": [
                255,
                0,
                100
            ]
    },
    {
        "id": 5,
        "name": "bus",
        "supercategory": "vehicle",
        "color": [
                0,
                60,
                100
            ]
    },
    {
        "id": 6,
        "name": "train",
        "supercategory": "vehicle",
        "color": [
                0,
                80,
                100
            ]
    },
    {
        "id": 7,
        "name": "truck",
        "supercategory": "vehicle",
        "color": [
                0,
                0,
                70
            ]
    },
    {
        "id": 8,
        "name": "boat",
        "supercategory": "vehicle",
        "color": [
                150,
                0,
                255
            ]
    },
    {
        "id": 9,
        "name": "traffic light",
        "supercategory": "outdoor",
        "color": [
                255,
                0,
                200
            ]
    },
    {
        "id": 10,
        "name": "fire hydrant",
        "supercategory": "outdoor",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 11,
        "name": "stop sign",
        "supercategory": "outdoor",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 12,
        "name": "parking meter",
        "supercategory": "outdoor",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 13,
        "name": "bench",
        "supercategory": "outdoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 14,
        "name": "bird",
        "supercategory": "animal",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 15,
        "name": "cat",
        "supercategory": "animal",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 16,
        "name": "dog",
        "supercategory": "animal",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 17,
        "name": "horse",
        "supercategory": "animal",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 18,
        "name": "sheep",
        "supercategory": "animal",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 19,
        "name": "cow",
        "supercategory": "animal",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 20,
        "name": "elephant",
        "supercategory": "animal",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 21,
        "name": "bear",
        "supercategory": "animal",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 22,
        "name": "zebra",
        "supercategory": "animal",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 23,
        "name": "giraffe",
        "supercategory": "animal",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 24,
        "name": "backpack",
        "supercategory": "accessory",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 25,
        "name": "umbrella",
        "supercategory": "accessory",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 26,
        "name": "handbag",
        "supercategory": "accessory",
        "color": [
                0,
                0,
                110
            ]
    },
        {
        "id": 27,
        "name": "tie",
        "supercategory": "accessory",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 28,
        "name": "suitcase",
        "supercategory": "accessory",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 29,
        "name": "frisbee",
        "supercategory": "sports",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 30,
        "name": "skis",
        "supercategory": "sports",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 31,
        "name": "snowboard",
        "supercategory": "sports",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 32,
        "name": "sports ball",
        "supercategory": "sports",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 33,
        "name": "kite",
        "supercategory": "sports",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 34,
        "name": "baseball bat",
        "supercategory": "sports",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 35,
        "name": "baseball glove",
        "supercategory": "sports",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 36,
        "name": "skateboard",
        "supercategory": "sports",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 37,
        "name": "surfboard",
        "supercategory": "sports",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 38,
        "name": "tennis racket",
        "supercategory": "sports",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 39,
        "name": "bottle",
        "supercategory": "kitchen",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 40,
        "name": "wine glass",
        "supercategory": "kitchen",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 41,
        "name": "cup",
        "supercategory": "kitchen",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 42,
        "name": "fork",
        "supercategory": "kitchen",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 43,
        "name": "knife",
        "supercategory": "kitchen",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 44,
        "name": "spoon",
        "supercategory": "kitchen",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 45,
        "name": "bowl",
        "supercategory": "kitchen",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 46,
        "name": "banana",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 47,
        "name": "apple",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 48,
        "name": "sandwich",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 49,
        "name": "orange",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 50,
        "name": "broccoli",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 51,
        "name": "carrot",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 52,
        "name": "hot dog",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 53,
        "name": "pizza",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 54,
        "name": "donut",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 55,
        "name": "cake",
        "supercategory": "food",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 56,
        "name": "chair",
        "supercategory": "furniture",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 57,
        "name": "couch",
        "supercategory": "furniture",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 58,
        "name": "potted plant",
        "supercategory": "furniture",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 59,
        "name": "bed",
        "supercategory": "furniture",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 60,
        "name": "dining table",
        "supercategory": "furniture",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 61,
        "name": "toilet",
        "supercategory": "furniture",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 62,
        "name": "tv",
        "supercategory": "electronic",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 63,
        "name": "laptop",
        "supercategory": "electronic",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 64,
        "name": "mouse",
        "supercategory": "electronic",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 65,
        "name": "remote",
        "supercategory": "electronic",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 66,
        "name": "keyboard",
        "supercategory": "electronic",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 67,
        "name": "cell phone",
        "supercategory": "electronic",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 68,
        "name": "microwave",
        "supercategory": "appliance",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 69,
        "name": "oven",
        "supercategory": "appliance",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 70,
        "name": "toaster",
        "supercategory": "appliance",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 71,
        "name": "sink",
        "supercategory": "appliance",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 72,
        "name": "refrigerator",
        "supercategory": "appliance",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 73,
        "name": "book",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 74,
        "name": "clock",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 75,
        "name": "vase",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 76,
        "name": "scissors",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 77,
        "name": "teddy bear",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 78,
        "name": "hair drier",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 79,
        "name": "toothbrush",
        "supercategory": "indoor",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 80,
        "name": "other",
        "supercategory": "other",
        "color": [
                0,
                0,
                110
            ]
    }
    
]




def split_to_coco_creator(input_instance_array, labels):
    labelid_matrix_name = []
    
    label_image_info = np.array(input_instance_array / 256, dtype=np.uint8)
    
    # Replacing values
    label_image_info[label_image_info < 30]=14
    label_image_info[(label_image_info > 34)&(label_image_info < 106)]=14
    label_image_info[(label_image_info > 115)]=14
    d = {30:0, 32:0, 33:0, 34:0, 106:8, 107:2, 108:2, 109:3, 110:3, 111:6, 112:2, 113:2, 114:7}
    for i,j in d.items():
        label_image_info[label_image_info==i] = j
        
    instance_image_info = np.array(input_instance_array % 256, dtype=np.uint8)

    unique_label_info = np.unique(label_image_info)
    unique_instance_info = np.unique(instance_image_info)

    for label_id, label in enumerate(labels):
        if (label_id in (unique_label_info)) and (label["instances"] == True):
            each_label_array = np.zeros(
                (input_instance_array.shape[0], input_instance_array.shape[1]),
                dtype=np.uint8,
            )

            each_label_array[label_image_info == label_id] = 255

            for instance_id in range(256):
                if instance_id in unique_instance_info:
                    each_instance_array = np.zeros(
                        (input_instance_array.shape[0],
                         input_instance_array.shape[1]),
                        dtype=np.uint8,
                    )

                    each_instance_array[instance_image_info ==
                                        instance_id] = 255

                    final_instance_array = np.bitwise_and(
                        each_instance_array, each_label_array
                    )

                    if np.unique(final_instance_array).size == 2:
                        labelid_matrix_name.append(
                            {
                                "label_id": label_id,
                                "instance_id": instance_id,
                                "label_name": label["readable"],
                                "image": final_instance_array,
                            }
                        )

    return labelid_matrix_name


def convert_class_id(annotation_filename):
    class_id = 14
    if "person" == annotation_filename:
        class_id = 0
    # elif "Bicyclist" == annotation_filename:
    #     class_id = 2
    # elif "Motorcyclist" == annotation_filename:
    #     class_id = 3
    # elif "Other Rider" == annotation_filename:
    #     class_id = 4
    elif "boat" == annotation_filename:
        class_id = 8
    elif "bus" == annotation_filename:
        class_id = 5
    elif "car" == annotation_filename:
        class_id = 2
    # elif "Caravan" == annotation_filename:
    #     class_id = 8
    elif "motorcycle" == annotation_filename:
        class_id = 3
    elif "train" == annotation_filename:
        class_id = 6
    # elif "Other Vehicle" == annotation_filename:
    #     class_id = 11
    # elif "Trailer" == annotation_filename:
    #     class_id = 12
    elif "truck" == annotation_filename:
        class_id = 7
    return class_id


def each_sub_proc(file_name, dir_name, dataset_root, image_id, labels, each_image_json):
    file_name = file_name[:-4]
    instance_path = "{}/{}/instances/{}.png".format(
        dataset_root, dir_name, file_name)
    instance_image = Image.open(instance_path)
    instance_array = np.array(instance_image, dtype=np.uint16)

    image_label_instance_infomatrix = split_to_coco_creator(
        instance_array, labels)

    image_info = pycococreatortools.create_image_info(
        image_id, file_name + ".jpg", instance_image.size
    )
    each_image_json["images"].append(image_info)

    segmentation_id = 1
    for item in image_label_instance_infomatrix:
        class_id = convert_class_id(item["label_name"])
        category_info = {"id": class_id, "is_crowd": 1}
        binary_mask = item["image"]

        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id,
            image_id,
            category_info,
            binary_mask,
            instance_image.size,
            tolerance=2,
        )
        if annotation_info is not None:
            each_image_json["annotations"].append(annotation_info)
            segmentation_id = segmentation_id + 1

    if each_image_json["images"] is []:
        print("Image {} doesn't contain one image".format(image_id))
    save_path = "{}/{}/massive_annotations/image{}_info.json".format(
        dataset_root, dir_name, image_id
    )
    print("Saving to {}".format(save_path))
    with open(save_path, "w") as fp:
        json.dump(each_image_json, fp)


def load_datasets_and_proc(dataset_root, dir_name, files, i):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    each_image_json = dict({"images": [], "annotations": []})

    if(FULL_CPU):
        pool = mp.Pool(os.cpu_count())
    else:
        pool = mp.Pool(os.cpu_count()-CPU_REDUCTION)

    with open("./config.json") as config_file:
        config = json.load(config_file)

    labels = config["labels"]
    for idx, image_filename in enumerate(files):
        pool.apply_async(
            each_sub_proc,
            args=(
                image_filename,
                dir_name,
                dataset_root,
                i + idx + 1,
                labels,
                each_image_json,
            ),
        )

    pool.close()
    pool.join()


def readout_each_image(dataset_root, dir_name, seq):
    json_saved_path = "{}/{}/massive_annotations/image{}_info.json".format(
        dataset_root, dir_name, seq
    )
    with open(json_saved_path) as fp:
        json_div = json.load(fp)

    return json_div


def main(dir_name, dataset_root, sample_type):
    dir_path = "{}/{}/instances".format(dataset_root, dir_name)
    files_list = []
    files = []
    i = 0
    for f in os.listdir(dir_path):
        if f.endswith("png"):
            files.append(f)
            i += 1
            print("Loading image {}: {}".format(i, f))
        if i > BATCH_SIZE//10 and i % BATCH_SIZE == 0:
            files_list.append(files)
            files = []
    if len(files) > 0:
        files_list.append(files)
        files = []

    # Pre-create needed image paths
    if (
        os.path.exists(
            "{}/{}/massive_annotations".format(dataset_root, dir_name))
        is False
    ):
        os.makedirs("{}/{}/massive_annotations".format(dataset_root, dir_name))

    batch = 0
    if not files_list:
        load_datasets_and_proc(dataset_root, dir_name, files)
    else:
        for files in files_list:
            batch+=1
            print("Batch n°{}".format(batch))
            if batch >= STARTING_BATCH:
                load_datasets_and_proc(dataset_root, dir_name, files, (batch-1)*BATCH_SIZE)

    combined_annotations = {
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    for idx in range(i):
        each_image_json = readout_each_image(dataset_root, dir_name, idx + 1)
        combined_annotations["images"].extend(each_image_json["images"])
        combined_annotations["annotations"].extend(
            each_image_json["annotations"])

    combined_annotations["annotations"] = sorted(
        combined_annotations["annotations"],
        key=lambda item: item.__getitem__("image_id"),
    )

    for idx in range(len(combined_annotations["annotations"])):
        combined_annotations["annotations"][idx]["id"] = idx + 1

    combined_json_path = "{}/{}/instances_shape_{}2020.json".format(
        dataset_root, dir_name, sample_type
    )
    with open(combined_json_path, "w") as fp:
        json.dump(combined_annotations, fp)


if __name__ == "__main__":
    dataset_root = ""
    dir_name_train = "MapillaryVistas/training/v2.0"
    dir_name_val = "MapillaryVistas/validation/v2.0"
    main(dir_name_train, dataset_root, "training")
    main(dir_name_val, dataset_root, "validation")