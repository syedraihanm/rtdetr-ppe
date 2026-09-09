
Construction Site Safety - v2 2023-11-26 4:12pm
==============================

This dataset was exported via roboflow.com on December 8, 2025 at 3:31 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 10875 images.
Personal_Protective_Equipment are annotated in YOLOv11 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 2 versions of each source image:
* Randomly crop between 0 and 30 percent of the image

The following transformations were applied to the bounding boxes of each image:
* Random Gaussian blur of between 0 and 2.5 pixels


