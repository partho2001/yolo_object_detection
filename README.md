# YOLO Object Detection with OpenCV

## Overview
This repository contains code for performing object detection using YOLO (You Only Look Once) with OpenCV. The code reads a video file, applies YOLO object detection to each frame, and outputs a new video with bounding boxes around detected objects.

## Files
- **yolo_object_detection.ipynb:** Jupyter Notebook containing the YOLO object detection code.
- **yolov3.weights:** Pre-trained YOLOv3 weights file.
- **yolov3.cfg:** YOLOv3 configuration file.
- **coco.names:** COCO dataset class names file.

## Instructions
1. **Download Pre-trained YOLO Weights and Configuration:**
    - Run the following commands in a code cell to download the required YOLO files:

    ```python
    !wget "https://pjreddie.com/media/files/yolov3.weights"
    !wget "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
    !wget "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    ```

2. **Run the YOLO Object Detection Code:**
    - Open and run the `yolo_object_detection.ipynb` notebook in a Jupyter environment.

3. **Input Video:**
    - Specify the path to the input video file in the `input_video_path` variable.

4. **Output Video:**
    - Specify the path to the output video file in the `output_video_path` variable.

5. **Execute the Code:**
    - Run the code cells in the notebook to perform YOLO object detection on the input video.

## Requirements
- Python 3.x
- OpenCV
- NumPy

## References
- YOLO: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- COCO Dataset: [https://cocodataset.org/](https://cocodataset.org/)


## INPUT FRAMES
![image](https://github.com/partho2001/yolo_object_detection/assets/42618752/ffd756f4-372d-4e4d-87ad-ba8beb5b000e)


## OUTPUT FRAMES
![image](https://github.com/partho2001/yolo_object_detection/assets/42618752/eb664c89-7a5f-42b1-ae7e-f979a9034333)
