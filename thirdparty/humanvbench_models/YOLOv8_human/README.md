YOLOv8 re-implementation for person detection using PyTorch

### Installation

```
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```

### Train

* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

| Version |                                                                              COCO weights |         CrowdHuman weights |
|:-------:|------------------------------------------------------------------------------------------:|---------------------------:|
|  v8_n   | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_n.pt) | [model](./weights/best.pt) |
|  v8_s   | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_s.pt) |                          - |
|  v8_m   | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_m.pt) |                          - |
|  v8_l   | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_l.pt) |                          - |
|  v8_x   | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_x.pt) |                          - |

* the weights are ported from original repo, see reference

### Dataset structure

    ├── Person 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

* Public person detection datasets
    * [COCO](https://cocodataset.org/#home)
    * [CrowdHuman](https://www.crowdhuman.org/download.html)
    * [HIEVE](http://humaninevents.org/data.html?title=1)
    * [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
    * [VFP290K](https://sites.google.com/view/dash-vfp300k/)
    * [Argoverse](https://eval.ai/web/challenges/challenge-page/800/overview)
    * [CEPDOF](https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/)
    * [HABBOF](https://vip.bu.edu/projects/vsns/cossy/datasets/habbof/)
    * [MW-R](https://vip.bu.edu/projects/vsns/cossy/datasets/mw-r/)
    * [TIDOS](https://vip.bu.edu/projects/vsns/cossy/datasets/tidos/)
    * [WEPDTOF](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/)
    * [DEPOF](https://vip.bu.edu/projects/vsns/cossy/datasets/depof/)
    * [FRIDA](https://vip.bu.edu/projects/vsns/cossy/datasets/frida/)

#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics
* https://github.com/open-mmlab/mmyolo
