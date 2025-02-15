import warnings
import torch

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/LWSDet.yaml')

    print(model.info())


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def format_parameters_count(count):
        if count < 1e6:
            return "{:.2f}K".format(count / 1e3)
        else:
            return "{:.2f}M".format(count / 1e6)


    # 计算参数量
    total_params1 = count_parameters(model)
    # 格式化并输出结果
    formatted_params1 = format_parameters_count(total_params1)
    # model.load("runs/train/yolov8n+BiFPN23/weights/best.pt")
    print("model Total Parameters: ", formatted_params1)
    model.train(data='ultralytics/cfg/datasets/vistrone.yaml',
                cache=False,
                imgsz=640,
                epochs=260,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # 如过想续训就设置last.pt的地址
                amp=True,  # 混合精度 如果出现训练损失为Nan可以关闭amp
                project='runs/train1',
                name='test',
                pretrained=False
                )

