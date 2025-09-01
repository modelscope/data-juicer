# image_pair_similarity_filter

Filter to keep image pairs with similarities between images within a specific range.

This operator uses a Hugging Face CLIP model to compute the cosine similarity between two images in each sample. It retains samples where the similarity score falls within the specified minimum and maximum thresholds. The 'any' strategy keeps a sample if any of the image pairs meet the condition, while the 'all' strategy requires all image pairs to meet the condition. The similarity scores are cached in the 'image_pair_similarity' field. Each sample must include exactly two distinct images.

过滤器将图像之间具有相似性的图像对保持在特定范围内。

该算子使用拥抱面部剪辑模型来计算每个样本中两个图像之间的余弦相似度。它保留相似性得分落在指定的最小和最大阈值内的样本。如果任何图像对满足条件，则 'any' 策略保持样本，而 'all' 策略要求所有图像对满足条件。相似性分数被缓存在 “image_pair_similarity” 字段中。每个样本必须恰好包含两个不同的图像。

Type 算子类型: **filter**

Tags 标签: cpu, hf, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_clip` |  | `'openai/clip-vit-base-patch32'` | clip model name on huggingface to compute |
| `trust_remote_code` |  | `False` |  |
| `min_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `0.1` | The min similarity to keep samples. |
| `max_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `1.0` | The max similarity to keep samples. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_pair_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_pair_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)