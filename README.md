# Object detection
Using deep learning models to detect objects from webcam input

## Object detection + voice feedback
Please watch the demo on Youtube https://youtu.be/W3x5il5w-jY

[![Demo](https://github.com/giangnn-bkace/object_detection/blob/master/images/demo.gif)](https://youtu.be/W3x5il5w-jY))


## Required Packages:
- tensorflow
- keras
- numpy
- opencv
- gtts
- pydub
- ffmpeg

## Some other result
(based on this repo: https://github.com/tensorflow/models/tree/master/research/object_detection)

### ssd_inception_v2_coco model

Speed: 42ms/frame   Coco mAP: 24 (higher is better)

![ssd](https://github.com/giangnn-bkace/object_detection/blob/master/images/ssd.gif)

### faster_rcnn_inception_v2_coco model

Speed: 58ms/frame   Coco mAP: 28

![rcnn](https://github.com/giangnn-bkace/object_detection/blob/master/images/faster_rcnn.gif)

### yolo_v3 model

Speed: 51ms/frame   Coco mAP: 33

![yolo](https://github.com/giangnn-bkace/object_detection/blob/master/images/yolo.gif)

