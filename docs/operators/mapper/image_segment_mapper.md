# image_segment_mapper

Perform segment-anything on images and return the bounding boxes.

This operator uses a FastSAM model to detect and segment objects in images, returning
their bounding boxes. It processes each image in the sample, and stores the bounding
boxes in the 'bbox_tag' field under the 'meta' key. If no images are present in the
sample, an empty array is stored instead. The operator allows setting the image
resolution, confidence threshold, and IoU (Intersection over Union) score threshold for
the segmentation process. Bounding boxes are represented as N x M x 4 arrays, where N is
the number of images, M is the number of detected boxes, and 4 represents the
coordinates.

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `imgsz` |  | `1024` | resolution for image resizing |
| `conf` |  | `0.05` | confidence score threshold |
| `iou` |  | `0.5` | IoU (Intersection over Union) score threshold |
| `model_path` |  | `'FastSAM-x.pt'` | the path to the FastSAM model. Model name should be |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_segment_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_segment_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)