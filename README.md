# Multi-Camera Multi-Person Tracking 
A multi-camera multi-person tracking system using YOLOv8 and DeepOCSORT. Includes additional features such as emotion recognition, gender recognition, and posture classification.

# Installation
1. Create a conda environment
    ```bash
    conda create -n mct python=3.8.17
    conda activate mct
    ```
2. Clone this repository and install necessary dependencies
    ```bash
    pip install -r requirements.txt 
    ```
3. To use CUDA, install `onnxruntime-gpu` based on your CUDA version as shown here: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

# Setup 
## Specifying Paths
- **Input videos**: To specify video sources, navigate to the `configs.yaml` file and locate the `sources` field. In this section, you can define multiple video sources that your application will process. (*Note: videos will be processed in parallel*)

    Example:
    ```yaml
    sources:
    - /path/to/video1.mp4
    - /path/to/video2.mp4
    - /path/to/video3.mp4
    ```
- **Output Location**: If you wish to customize the output location for the processed data, find the `output_location` entry within the `configs.yaml` file.

    Example:
    ```yaml
    output_location: /path/to/output/folder
    ```

## Modifying Hyperparameters
Within the `configs.yaml` file, you have the flexibility to adjust essential hyperparameters that influence the behavior of your application. Three key hyperparameters, namely `tracker_configs`, `person_seg_configs`, and `face_det_configs`, are defined in this configuration file.

- **Tracker Configuration**: The `tracker_configs` section allows you to fine-tune parameters associated with the tracking module. Depending on your specific requirements, you can experiment with values such as `reid_thresh`, `assoc_func`, `iou_thresh`, etc.

- **Person Segmentation and Face Detection Configuration**: The `person_seg_configs` and `face_det_configs` sections correspond to modules trained using Ultralytics. Refer to their documentation for detailed parameter information.

## Anonymizing Output Videos
You can choose to anonymize the people by setting `anonymize` to `true` in `configs.yaml`. Enabling the this parameter ensures that each detected person is anonymized by applying a black mask.

# Usage
Once you have completed the setup process and configured your parameters as desired, run the script:
```python
python main.py --configs configs.yaml
```

# Models and Datasets
| **Description** | **Model** | **Dataset** | **Note** |
|:---:|:---:|:---:|:---:|
| Emotion Recognition | [mini-Xception](https://github.com/oarriaga/face_classification) | [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) | converted to ONNX from original source |
| Gender Recognition | [mini-Xception](https://github.com/oarriaga/face_classification) | [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) | converted to ONNX from original source |
| Object Detection & Segmentation | [YOLOv8m-seg](https://github.com/ultralytics/ultralytics) | [COCO](https://cocodataset.org) | extra training with more augmentations |
| Face Detection | [YOLOv8n](https://github.com/derronqi/yolov8-face) | [WIDER Face](http://shuoyang1213.me/WIDERFACE/) | taken from original source |
| Posture Classification | [YOLOv8x-cls](https://github.com/ultralytics/ultralytics) | [POLAR](https://data.mendeley.com/datasets/hvnsh7rwz7/1) | custom trained with only 3 classes |
| Person Reidentification | [YoutuReid](https://github.com/ReID-Team/ReID_extra_testdata) | DukeMTMC | taken from original source |

# References
- https://github.com/oarriaga/face_classification
- https://github.com/mikel-brostrom/yolo_tracking
- https://github.com/ultralytics/ultralytics
- https://github.com/derronqi/yolov8-face
- https://github.com/ReID-Team/ReID_extra_testdata
- https://data.mendeley.com/datasets/hvnsh7rwz7/1