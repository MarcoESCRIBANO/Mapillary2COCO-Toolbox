# Mapillary2COCO-Toolbox

**Research-focused dataset transformation toolbox**  
Developed to support research on **instance segmentation for embedded real-time object detection**. This toolbox enables converting Mapillary Vistas annotations to COCO format and resizing datasets for model training.

## Objective
Facilitate reproducible experiments by transforming Mapillary Vistas instance segmentation data into COCO format, with flexible class selection and resizing support. Supports research on improving real-time object detection on embedded systems.

## Method
- Conversion of Mapillary instance annotations to COCO format  
- Resizing of images and segmentation masks for model compatibility  
- Visualization of masks and bounding boxes for verification  

## Implementation
- Python scripts (`main.py`, `resize.py`, `showAnnotation.py`) handle conversion, resizing, and visualization  
- Docker-based setup for reproducibility  
- Supports selective class conversion or full dataset processing  

## Research Context & Contribution
This toolbox was developed as part of research titled **“Instance Segmentation for Embedded Real-Time Object Detector Improvement”** (Multitel R&D).  
It enabled experiments exploring the potential of instance segmentation to improve real-time object detection on embedded platforms. While the approach tested did not lead to performance gains with the segmentation methods used, the toolbox provided a reproducible framework for further experimentation.
