from PIL import Image
import os
import multiprocessing as mp
import numpy as np
import json
import shutil
from pycocotools.coco import COCO
from pycocotools import mask


def resize_Images(dir_path, saving_path, file_type):
    files_image = []
    i = 0
    for f in os.listdir(dir_path):
        if f.endswith(file_type):
            files_image.append(f)
            i += 1
            
    # Pre-create needed image paths
    if (
        os.path.exists(saving_path)
        is False
    ):
        os.makedirs(saving_path)

    for idx, image_filename in enumerate(files_image):
        image_tmp = Image.open(os.path.join(dir_path, image_filename))
        w, h = image_tmp.size
        ratio = w/h
        down_w = int(550)
        down_h = int(550//ratio)
        down_points = (down_w, down_h)
        image_resized = image_tmp.resize(down_points,Image.BICUBIC)
        image_resized = image_resized.save(os.path.join(saving_path, image_filename))
        

def mapCatToCOCOCat(cat_id):
    if cat_id == 1:
        return 1
    elif cat_id == 2:
        return 1
    elif cat_id == 3:
        return 1
    elif cat_id == 4:
        return 1
    elif cat_id == 5:
        return 9
    elif cat_id == 6:
        return 6
    elif cat_id == 7:
        return 3
    elif cat_id == 8:
        return 3
    elif cat_id == 9:
        return 4
    elif cat_id == 10:
        return 7
    elif cat_id == 11:
        return 3
    elif cat_id == 12:
        return 3
    elif cat_id == 13:
        return 8
    else:
        return -1
    


def resize_Annotations(dir_path, saving_path, file_name):
    # Pre-create needed file paths
    if (
        os.path.exists(saving_path)
        is False
    ):
        os.makedirs(saving_path)
        
    dir_path = "{}/{}".format(dir_path, file_name)
    saving_path = "{}/{}".format(saving_path, file_name)
    shutil.copyfile(dir_path, saving_path)
        
    Map = COCO(dir_path)
    
    ann_ids = Map.getAnnIds(iscrowd=True)
    anns = Map.loadAnns(ann_ids)
    last_image_id = -1
    
    for ann in anns:
        segm = ann['segmentation']
        ann_id = ann['id']
        image_id = ann["image_id"]
        
        # For coco eval on yolact
        cat_id = ann["category_id"]
        cat_id = mapCatToCOCOCat(cat_id)
        
        assert sum(segm['counts']) == segm['size'][0] * segm['size'][1]

        # Draw RLE label
        label = np.zeros(segm['size'], np.uint8).reshape(-1)
        ids = 0
        value = 0
        for c in segm['counts']:
            label[ids: ids+c] = value
            value = not value
            ids += c
        
        label = label.reshape(segm['size'], order='F') # order='F' means Fortran memory order
        
        image_tmp = Image.fromarray(label*255)
        w, h = image_tmp.size
        ratio = w/h
        down_w = int(550)
        down_h = int(550//ratio)
        down_points = (down_w, down_h)
        image_resized = image_tmp.resize(down_points,Image.BICUBIC)
        image_resized_binary = np.array(image_resized, dtype=np.uint8)
        image_resized = image_resized_binary.reshape(down_w*down_h, order='F')
        image_resized = image_resized//255
        
        count = []
        value = 0
        counter = -1
        for id in image_resized:
            if id == value:
                counter+=1
            else:
                counter += 1
                count.append(counter)
                value = not value
                counter = 0
        if counter != 0:
            counter += 1
            count.append(counter)
            
        assert sum(count) == down_w * down_h
        
        with open(saving_path, 'r+') as f:
            data = json.load(f)
            data['annotations'][ann_id-1]['segmentation']['counts'] = count 
            data['annotations'][ann_id-1]['segmentation']['size'] = [down_h, down_w]
            data['annotations'][ann_id-1]['width'] = down_w
            data['annotations'][ann_id-1]['height'] = down_h
            bbox = data['annotations'][ann_id-1]['bbox']
            data['annotations'][ann_id-1]['bbox'] = [round(bbox[0]*down_w/w,1), round(bbox[1]*down_h/h,1), round(bbox[2]*down_w/w,1), round(bbox[3]*down_h/h,1)]
            data['annotations'][ann_id-1]['area'] = int(mask.area(mask.encode(np.asfortranarray(image_resized_binary.astype(np.uint8)))))
            
            # For coco eval on yolact
            data['annotations'][ann_id-1]['category_id'] = cat_id
            data['annotations'][ann_id-1]['iscrowd'] = 0
            
            if  image_id != last_image_id:
                data['images'][image_id-1]['width'] = down_w
                data['images'][image_id-1]['height'] = down_h
                last_image_id = image_id  
            f.seek(0)  
            json.dump(data, f)
            f.truncate() 
         

def main(dir_name_train, dir_name_val, dataset_root):
    pool = mp.Pool(os.cpu_count()-4)
    
    # 
    # Training data
    # 
    dir_image_path = "{}/{}/images".format(dataset_root, dir_name_train)
    # dir_instance_path = "{}/{}/v2.0/instances".format(dataset_root, dir_name_train)
    dir_annotation_path = "{}/{}/v2.0".format(dataset_root, dir_name_train)
    
    saving_image_path = "{}Resized/{}/images".format(dataset_root, dir_name_train)
    # saving_instance_path = "{}Resized/{}/v2.0/instances".format(dataset_root, dir_name_train)
    saving_annotation_path = "{}Resized/{}/v2.0".format(dataset_root, dir_name_train)
    
    pool.apply_async(resize_Images,args=(dir_image_path, saving_image_path, "jpg"))
    # pool.apply_async(resize_Images,args=(dir_instance_path, saving_instance_path, "png"))
    pool.apply_async(resize_Annotations,args=(dir_annotation_path, saving_annotation_path, "instances_shape_training2020.json"))
    # resize_Annotations(dir_annotation_path, saving_annotation_path, "instances_shape_training2020.json")
    
    # 
    # Validation data
    #     
    dir_image_path = "{}/{}/images".format(dataset_root, dir_name_val)
    # dir_instance_path = "{}/{}/v2.0/instances".format(dataset_root, dir_name_val)
    dir_annotation_path = "{}/{}/v2.0".format(dataset_root, dir_name_val)
    
    saving_image_path = "{}Resized/{}/images".format(dataset_root, dir_name_val)
    # saving_instance_path = "{}Resized/{}/v2.0/instances".format(dataset_root, dir_name_val)
    saving_annotation_path = "{}Resized/{}/v2.0".format(dataset_root, dir_name_val)
    
    pool.apply_async(resize_Images,args=(dir_image_path, saving_image_path, "jpg"))
    # pool.apply_async(resize_Images,args=(dir_instance_path, saving_instance_path, "png"))
    pool.apply_async(resize_Annotations,args=(dir_annotation_path, saving_annotation_path, "instances_shape_validation2020.json"))

    pool.close()
    pool.join()

if __name__ == "__main__":
    dataset_root = "MapillaryVistas"
    dir_name_train = "training"
    dir_name_val = "validation"
    main(dir_name_train, dir_name_val, dataset_root)
