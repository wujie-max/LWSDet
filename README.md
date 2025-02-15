# Lightweight Adaptive Feature Selection Network for Enhanced Small Object Detection in UAV Imagery

Official PyTorch implementation of **LWSDet**.

[Lightweight Adaptive Feature Selection Network for Enhanced Small Object Detection in UAV Imagery]
Jinxia Yu, Jie Wu, Yongli Tang


<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
To address the challenges of excessive parameters and limited detection accuracy in small object detection algorithms for unmanned aerial vehicles (UAVs), this paper introduces a novel lightweight and efficient model termed LWSDet. Firstly, an Adaptive Feature Selection (AFS) module is proposed, leveraging a gating mechanism to dynamically weight input features and employing depthwise convolution to extract multi-scale information. This module enhances feature extraction accuracy while reducing the model's parameter count by focusing adaptively on key features. Secondly, a Multi-Layer Feature Fusion (MLFF) module is designed, efficiently integrating multi-layer feature information through techniques such as point convolution, upsampling, and downsampling. By fusing shallow detail information with deep semantic information, the MLFF module significantly improves the algorithm's robustness and generalization in complex scenarios. Finally, a Partial Ghost Convolution (PGConv) module is presented, combining the concepts of ghost convolution and partial convolution to optimize the standard convolution process. This module generates feature maps comparable to standard convolution through linear mapping while retaining part of the original feature maps to supplement detailed information, thereby enriching feature representation and effectively reducing computational overhead. Experiments conducted on three datasets demonstrate that LWSDet achieves a notable improvement in mean Average Precision (mAP50) by 2.9% and a reduction in parameters by 36.8% compared to the baseline model, highlighting its effectiveness and efficiency for small object detection in UAV images. 
</details>

## DataSets

Vistrone2019
(https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip,
          https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip,
          https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip)

It is important to include a script in ultralytics/cfg/datasets/VisDrone. yaml that converts Vistrone2019 to YOLO format.

## Installation
`conda` virtual environment is recommended. 
```
conda create -n LWSDet python=3.10
conda activate LWSDet
pip install -r requirements.txt
pip install -e .
```
## Performance

Vistrone2019

| Model          | Test Size | #Params | mAP50(%) |  mAP50:95(%)   | 
|:---------------| :-------: |:-------:|:--------:|:--------------:|
| RetinaNet[1]   |    640    | 37.96M  |   41.6   |      24.1      |
| LODNU[2]       |    640    |  8.71M  |   25.9   |      17.3      |
| LAI-YOLOv5s[3] |    640    |  **6.3M**   |   40.4   |       -        | 
| YOLOv5s[4]     |    640    |  61.6M  |   39.4   |      22.4      | 
| YOLOv5m[5]     |    640    | 20.97M  |   36.3   |      20.5      | 
| YOLOv10s[6]    |    640    |  7.93M  |   38.6   |       23       |
| LWSDet(Our)    |    640    |  7.07M  |   **41.5**   |  **24.4**      |

## Training 
```
python train.py
```

## Validation
Note that a smaller confidence threshold can be set to detect smaller objects or objects in the distance. 
```
python val.py
```


## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).

Thanks for the great implementations! 

## Citation
If our code or models help your work, please cite our paper:

```BibTeX
Our article has not been published yet, we will update it once it is published.
```
References:

[1] Lin T. Focal Loss for Dense Object Detection[J]. arXiv preprint arXiv:1708.02002, 2017.

[2] Chen N, Li Y, Yang Z, et al. LODNU: lightweight object detection network in UAV vision[J]. The Journal of Supercomputing, 2023, 79(9): 10117-10138.

[3] Deng L, Bi L, Li H, et al. Lightweight aerial image object detection algorithm based on improved YOLOv5s[J]. Scientific reports, 2023, 13(1): 7817.

[4] A. Farhadi and J. Redmon. Yolov3: An incremental improvement. Computer vision and pattern recognition. 1804, 1–6(2018).

[5] G. Jocher. Ultralytics YOLOv5 version 7.0. url: https://github.com/ultralytics/yolov5(2020)

[6] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding. Yolov10: Real-time end-to-end object detection. arXiv preprint arXiv:2405.14458.(2024).


