import pycocotools.mask as mask
from pycocotools.coco import COCO
import numpy as np
import cv2
import operator
import os

map_dataset_path = "/MapillaryVistas/"
# map_dataset_path = "/MapillaryVistasResized/"

train_val_path = "training"
# train_val_path = "validation"

Map = COCO(map_dataset_path + train_val_path + "/v2.0/instances_shape_" + train_val_path + "2020.json")


ann_ids = Map.getAnnIds(iscrowd=True)
anns = Map.loadAnns(ann_ids)
cats = Map.cats
anns1 = [ann for ann in anns if len(ann['segmentation']) > 1]
last_image_id = -1

for ann in anns:
    image_id = ann["image_id"]
    segm = ann['segmentation']
    bbox = ann['bbox']
    cat_id = ann['category_id']
    color = cats[cat_id]["color"]
    
    bbox = np.array(ann["bbox"])
    bbox[2:4] = bbox[0:2] + bbox[2:4]
    
    temp_image_path = map_dataset_path + "RLE_label.png"
    
    assert sum(segm['counts']) == segm['size'][0] * segm['size'][1]

    # Draw RLE label
    label = np.zeros(segm['size'], np.uint8).reshape(-1)
    ids = 0
    value = 0
    for c in segm['counts']:
        label[ids: ids+c] = value
        value = not value
        ids += c
    
    label = label.reshape(segm['size'], order='F')
    
    cv2.imwrite(temp_image_path, label*255)

    image_info = Map.loadImgs(image_id)
    image_path = image_info[0]["file_name"]
    image_path = map_dataset_path + train_val_path + "/images/" + image_path
    image = cv2.imread(image_path)
    save_labeled_image_path = map_dataset_path + "Labelled_" + train_val_path + "_image"+ str(image_id) + ".png"
    
    if (os.path.isfile(save_labeled_image_path) is False) or (image_id != last_image_id):
        cv2.imwrite(save_labeled_image_path, image)
    last_image_id = image_id  
        
    image_overlay = cv2.imread(temp_image_path)
    
    overlay = image_overlay.copy()
    h, w, c = overlay.shape
    x, y, w, h = 0, 0, w, h
    overlay = cv2.rectangle(overlay, (x, y), (x+w, y+h), tuple(map(operator.sub, (255,255,255), tuple(color))), -1) 
    overlay = cv2.subtract(image_overlay, overlay)
   
    background = cv2.imread(save_labeled_image_path)
    added_image = cv2.addWeighted(background, 1, overlay, 0.75, 0)
    
    cv2.rectangle(added_image, (int(bbox[0]), int(bbox[1])), 
                  (int(bbox[2]), int(bbox[3])), tuple(color), 1)
    
    cv2.imwrite(save_labeled_image_path, added_image)


      