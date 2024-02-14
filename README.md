# Mapillary dataset tool box


## Table of Contents

* [Mapillary to COCO annotations](#Convert-Mapillary-Vistas-Dataset-to-Coco-annotations-format)
* [Resize images and COCO annotations](#Resize-images-and-COCO-annotations)
* [Show COCO format mask and bbox on image](#Show-COCO-format-mask-and-bbox-on-image)

## Convert Mapillary Vistas Dataset to Coco annotations format

>main.py

This python script helps you convert your mapillary vistas dataset to coco format and choose the classes you want to keep.

#### Brief Introduction

Based on instance images in Mapillary vistas dataset.
In this `instance` image, label info is embeded into each pixel value.

```python
pixel / 256 # the value represents this pixel belongs to which label.
pixel % 256 # the value represents this pixel is the i-th instance of its label.
```

## Resize images and COCO annotations

>resize.py

This python script helps you resize your transformed mapillary vistas datasets to fit your model.
It can also match your classes with corresponding COCO classes to evaluate your dataset on models pre-trained on COCO.

#### Brief Introduction

Rewrite your Json annotations with resized segmentation mask, bbox and image information.
Resize your dataset images
Can also rewrite your categorie_id (classe)

## Show COCO format mask and bbox on image

>ShowMapAnnotation.py

This python script helps you visualize your dataset mask and bbox.