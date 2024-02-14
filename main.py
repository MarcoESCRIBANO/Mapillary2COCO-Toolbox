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

FULL_CPU = False

INFO = {
    "description": "Mapillary",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "Luodian",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    }
]

CATEGORIES = [
    {
        "id": 1,
        "name": "Person",
        "supercategory": "human--person--individual",
        "color": [
                220,
                20,
                60
            ]
    },
    {
        "id": 2,
        "name": "Bicyclist",
        "supercategory": "human--rider--bicyclist",
        "color": [
                255,
                0,
                0
            ]
    },
    {
        "id": 3,
        "name": "Motorcyclist",
        "supercategory": "human--rider--motorcyclist",
        "color": [
                255,
                0,
                100
            ]
    },
    {
        "id": 4,
        "name": "Other_Rider",
        "supercategory": "human--rider--other-rider",
        "color": [
                255,
                0,
                200
            ]
    },
    {
        "id": 5,
        "name": "Boat",
        "supercategory": "object--vehicle--boat",
        "color": [
                150,
                0,
                255
            ]
    },
    {
        "id": 6,
        "name": "Bus",
        "supercategory": "object--vehicle--bus",
        "color": [
                0,
                60,
                100
            ]
    },
    {
        "id": 7,
        "name": "Car",
        "supercategory": "object--vehicle--car",
        "color": [
                0,
                0,
                142
            ]
    },
    {
        "id": 8,
        "name": "Caravan",
        "supercategory": "object--vehicle--caravan",
        "color": [
                0,
                0,
                90
            ]
    },
    {
        "id": 9,
        "name": "Motorcycle",
        "supercategory": "object--vehicle--motorcycle",
        "color": [
                0,
                0,
                230
            ]
    },
    {
        "id": 10,
        "name": "On_Rails",
        "supercategory": "object--vehicle--on-rails",
        "color": [
                0,
                80,
                100
            ]
    },
    {
        "id": 11,
        "name": "Other_Vehicle",
        "supercategory": "object--vehicle--other-vehicle",
        "color": [
                128,
                64,
                64
            ]
    },
    {
        "id": 12,
        "name": "Trailer",
        "supercategory": "object--vehicle--trailer",
        "color": [
                0,
                0,
                110
            ]
    },
    {
        "id": 13,
        "name": "Truck",
        "supercategory": "object--vehicle--truck",
        "color": [
                0,
                0,
                70
            ]
    }
    
]


def split_to_coco_creator(input_instance_array, labels):
    labelid_matrix_name = []
    
    label_image_info = np.array(input_instance_array / 256, dtype=np.uint8)
    
    # Replacing values
    label_image_info[label_image_info < 30]=14
    label_image_info[(label_image_info > 34)&(label_image_info < 105)]=14
    label_image_info[(label_image_info > 115)]=14
    d = {30:0, 32:1, 33:2, 34:3, 105:4, 106:14, 107:5, 108:6, 109:7, 110:8, 111:9, 112:10, 113:11, 114:12}
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
    class_id = 0
    if "Person" == annotation_filename:
        class_id = 1
    elif "Bicyclist" == annotation_filename:
        class_id = 2
    elif "Motorcyclist" == annotation_filename:
        class_id = 3
    elif "Other Rider" == annotation_filename:
        class_id = 4
    elif "Boat" == annotation_filename:
        class_id = 5
    elif "Bus" == annotation_filename:
        class_id = 6
    elif "Car" == annotation_filename:
        class_id = 7
    elif "Caravan" == annotation_filename:
        class_id = 8
    elif "Motorcycle" == annotation_filename:
        class_id = 9
    elif "On Rails" == annotation_filename:
        class_id = 10
    elif "Other Vehicle" == annotation_filename:
        class_id = 11
    elif "Trailer" == annotation_filename:
        class_id = 12
    elif "Truck" == annotation_filename:
        class_id = 13
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


def load_datasets_and_proc(dataset_root, dir_name, files):
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    each_image_json = dict({"images": [], "annotations": []})

    if(FULL_CPU):
        pool = mp.Pool(os.cpu_count())
    else:
        pool = mp.Pool(os.cpu_count()-4)

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
                idx + 1,
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
        if i > 100 and i % 600 == 1:
            files_list.append(files)
            files = []
        if f.endswith("png"):
            files.append(f)
            i += 1
            print("Loading image {}: {}".format(i, f))

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
            if batch > 0:
                load_datasets_and_proc(dataset_root, dir_name, files)

    combined_annotations = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    for idx in range(int(len(files))):
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
