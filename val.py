import time
import warnings
import  torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import fuse_conv_and_bn
from ultralytics.nn.Addmodules import *
from ultralytics import YOLOv10

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLOv10('runs/train1/yolov8s-PT(VOC07+12)/weights/best.pt')

    model.val(
        data='ultralytics/cfg/datasets/VOC.yaml',
        batch = 1,
        imgsz=512,
        project='runs/val',
        name='yolov8s-PT(VOC07+12)',
    )
    # print(model.info())

