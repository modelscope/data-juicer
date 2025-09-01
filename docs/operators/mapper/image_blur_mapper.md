# image_blur_mapper

Blurs images in the dataset with a specified probability and blur type.

This operator blurs images using one of three types: mean, box, or Gaussian. The probability of an image being blurred is controlled by the `p` parameter. The blur effect is applied using a kernel with a specified radius. Blurred images are saved to a directory, which can be specified or defaults to the input directory. If the save directory is not provided, the `DJ_PRODUCED_DATA_DIR` environment variable can be used to set it. The operator ensures that the blur type is one of the supported options and that the radius is non-negative.

使用指定的概率和模糊类型对数据集中的图像进行模糊处理。

此运算符使用以下三种类型之一来模糊图像: 均值，框或高斯。图像被模糊的概率由 'p' 参数控制。使用具有指定半径的内核应用模糊效果。模糊图像将保存到一个目录中，该目录可以指定或默认为输入目录。如果未提供保存目录，则可以使用 'dj_producted_data_dir' 环境变量来设置它。该运算符确保 “模糊类型” 是受支持的选项之一，并且半径为非负。

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `p` | <class 'float'> | `0.2` | Probability of the image being blurred. |
| `blur_type` | <class 'str'> | `'gaussian'` | Type of blur kernel, including |
| `radius` | <class 'float'> | `2` | Radius of blur kernel. |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_blur_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_blur_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)