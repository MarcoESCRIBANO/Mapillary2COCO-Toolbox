from PIL import Image
import os
import multiprocessing as mp
import numpy as np
import shutil
from pycocotools.coco import COCO
from pycocotools import mask
import pycococreatortools
import ujson

### Set parameters ###

NEW_SIZE = 550

def resize(dir_image_path, saving_image_path,dir_annotation_path, saving_annotation_path, val_train):        
    # Pre-create needed image paths
    if (
        os.path.exists(saving_image_path)
        is False
    ):
        os.makedirs(saving_image_path)

    # Pre-create needed file paths
    if (
        os.path.exists(saving_annotation_path)
        is False
    ):
        os.makedirs(saving_annotation_path)
        
    dir_path = "{}/{}".format(dir_annotation_path, val_train)
    saving_path = "{}/{}".format(saving_annotation_path, val_train)
    shutil.copyfile(dir_path, saving_path)
        
    Map = COCO(dir_path)
    with open(saving_path, 'r+') as f:
        data = ujson.load(f)
        # Resize images
        for idx, image in Map.imgs.items():
            image_tmp = Image.open(os.path.join(dir_image_path, image["file_name"]))
            w, h = image_tmp.size
            ratio = w/h
            down_w = int(NEW_SIZE)
            down_h = int(NEW_SIZE//ratio)
            down_points = (down_w, down_h)
            image_resized = image_tmp.resize(down_points,Image.BICUBIC)
            image_resized = image_resized.save(os.path.join(saving_image_path, image["file_name"]))
            
            data['images'][idx-1]['width'] = down_w
            data['images'][idx-1]['height'] = down_h

        # Resize annotations where is is_crowd is True
        ann_ids = Map.getAnnIds(iscrowd=True)
        anns = Map.loadAnns(ann_ids)
        for ann in anns:
            ann_id = ann['id']
            image_id = ann['image_id']
            down_w = data['images'][image_id-1]['width']
            down_h = data['images'][image_id-1]['height']
            image = data['images'][image_id-1]
            image_tmp = Image.open(os.path.join(dir_image_path, image["file_name"]))
            w, h = image_tmp.size
            bbox = ann["bbox"]
            Bmask = Map.annToMask(ann)
            down_points = (down_w, down_h)

            Bmask = pycococreatortools.resize_binary_mask(Bmask,down_points)
            binary_mask_encoded = mask.encode(np.asfortranarray(Bmask.astype(np.uint8)))

            area = mask.area(binary_mask_encoded)
            rle = pycococreatortools.binary_mask_to_rle(Bmask)
            
            data['annotations'][ann_id-1]['segmentation'] = rle
            data['annotations'][ann_id-1]['area'] = int(area)

            data['annotations'][ann_id-1]['width'] = down_w
            data['annotations'][ann_id-1]['height'] = down_h
            data['annotations'][ann_id-1]['bbox'] = [round(bbox[0]*down_w/w,1), round(bbox[1]*down_h/h,1), round(bbox[2]*down_w/w,1), round(bbox[3]*down_h/h,1)]

        # Resize annotations where is is_crowd is False
        ann_ids = Map.getAnnIds(iscrowd=False)
        anns = Map.loadAnns(ann_ids)
        for ann in anns:
            ann_id = ann['id']
            image_id = ann['image_id']
            down_w = data['images'][image_id-1]['width']
            down_h = data['images'][image_id-1]['height']
            image = data['images'][image_id-1]
            image_tmp = Image.open(os.path.join(dir_image_path, image["file_name"]))
            w, h = image_tmp.size
            bbox = ann["bbox"]
            Bmask = Map.annToMask(ann)
            down_points = (down_w, down_h)

            Bmask = pycococreatortools.resize_binary_mask(Bmask,down_points)
            binary_mask_encoded = mask.encode(np.asfortranarray(Bmask.astype(np.uint8)))

            area = mask.area(binary_mask_encoded)
            poly = pycococreatortools.binary_mask_to_polygon(Bmask,2)

            data['annotations'][ann_id-1]['segmentation'] = poly 
            if data['annotations'][ann_id-1]['segmentation'] == []:
                rle = pycococreatortools.binary_mask_to_rle(Bmask)
                data['annotations'][ann_id-1]['segmentation'] = rle
                data['annotations'][ann_id-1]['iscrowd'] = 1
            data['annotations'][ann_id-1]['area'] = int(area)

            data['annotations'][ann_id-1]['width'] = down_w
            data['annotations'][ann_id-1]['height'] = down_h
            data['annotations'][ann_id-1]['bbox'] = [round(bbox[0]*down_w/w,1), round(bbox[1]*down_h/h,1), round(bbox[2]*down_w/w,1), round(bbox[3]*down_h/h,1)]


        # Save modification in the COCO annotation JSON        
        f.seek(0)  
        ujson.dump(data, f)
        f.truncate() 

def main(dir_name_train, dir_name_val, dataset_root):
    pool = mp.Pool(os.cpu_count()-4)
    
    # 
    # Training data
    # 
    dir_image_path = "{}/{}/images".format(dataset_root, dir_name_train)
    dir_annotation_path = "{}/{}/v2.0".format(dataset_root, dir_name_train)
    
    saving_image_path = "{}Resized/{}/images".format(dataset_root, dir_name_train)
    saving_annotation_path = "{}Resized/{}/v2.0".format(dataset_root, dir_name_train)

    pool.apply_async(resize,args=(dir_image_path, saving_image_path,dir_annotation_path, saving_annotation_path, "instances_shape_training2020.json"))
    # resize(dir_image_path, saving_image_path,dir_annotation_path, saving_annotation_path, "instances_shape_training2020.json")


    # 
    # Validation data
    #     
    dir_image_path = "{}/{}/images".format(dataset_root, dir_name_val)
    dir_annotation_path = "{}/{}/v2.0".format(dataset_root, dir_name_val)
    
    saving_image_path = "{}Resized/{}/images".format(dataset_root, dir_name_val)
    saving_annotation_path = "{}Resized/{}/v2.0".format(dataset_root, dir_name_val)
    
    pool.apply_async(resize,args=(dir_image_path, saving_image_path,dir_annotation_path, saving_annotation_path, "instances_shape_validation2020.json"))
    # resize(dir_image_path, saving_image_path,dir_annotation_path, saving_annotation_path, "instances_shape_validation2020.json")

    pool.close()
    pool.join()

if __name__ == "__main__":
    dataset_root = "MapillaryVistas"
    dir_name_train = "training"
    dir_name_val = "validation"
    main(dir_name_train, dir_name_val, dataset_root)