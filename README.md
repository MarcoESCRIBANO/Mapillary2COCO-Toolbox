# Mapillary dataset toolbox


## Table of Contents

* [Build and Run](#Build-and-Run)
* [Mapillary to COCO annotations](#Convert-Mapillary-Vistas-Dataset-to-Coco-annotations-format)
* [Resize images and COCO annotations](#Resize-images-and-COCO-annotations)
* [Show COCO format mask and bbox on image](#Show-COCO-format-mask-and-bbox-on-image)
* [Sources](#Sources)


## Getting Started

>compose.yaml

Put the dataset folder path you want to work with in volumes.

>Dockerfile

Uncomment the command you want to execute (ex: ENTRYPOINT [ "python", "/main.py" ] -> convert to COCO)

Build and run:
```zsh
# Build image, run and save log into a log file
docker compose up --build &> run.log 
```
```zsh
# Build image and run in background
docker compose up --build -d 
```


## Convert Mapillary Vistas Dataset to Coco annotations format

>main.py

This python script helps you convert your mapillary vistas dataset to coco format and choose the classes you want to keep. (Use main_full_labels.py if you want to convert all Mapillary dataset classes)

#### Brief Introduction

Based on instance images in Mapillary vistas dataset.
In this `instance` image, label info is embeded into each pixel value.

```python
pixel / 256 # this value represents the label that this pixel belongs to.
pixel % 256 # this value represents that this pixel is the i-th instance of its label.
```


## Resize images and COCO annotations

>resize.py

This python script helps you resize the images in your transformed mapillary vistas datasets annotations to fit your model.

#### Brief Introduction

Put the target folder path (where you want your resized dataset) in the compose.yaml file volumes. \
The algorithm rewrite your Json annotations with resized segmentation mask, bbox and image information and also resize your dataset images.



## Show COCO format mask and bbox on image

>showAnnotation.py

This python script helps you visualize your dataset mask and bbox.

## Sources

* [Mapillary2COCO](https://github.com/Luodian/Mapillary2COCO) repository from [Luodian](https://github.com/Luodian)
* [show-coco-annos.py](https://gist.github.com/tangh/0d398813dd3e64a72d830149c0363742) from [tangh](https://gist.github.com/tangh)

