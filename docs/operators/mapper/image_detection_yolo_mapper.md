# image_detection_yolo_mapper

Perform object detection using YOLO on images and return bounding boxes and class
labels.

This operator uses a YOLO model to detect objects in images. It processes each image in
the sample, returning the bounding boxes and class labels for detected objects. The
operator sets the `bbox_tag` and `class_label_tag` fields in the sample's metadata. If
no image is present or no objects are detected, it sets `bbox_tag` to an empty array and
`class_label_tag` to -1. The operator uses a confidence score threshold and IoU
(Intersection over Union) score threshold to filter detections.

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `imgsz` |  | `640` | resolution for image resizing |
| `conf` |  | `0.05` | confidence score threshold |
| `iou` |  | `0.5` | IoU (Intersection over Union) score threshold |
| `model_path` |  | `'yolo11n.pt'` | the path to the YOLO model. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_detection_yolo_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_detection_yolo_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)