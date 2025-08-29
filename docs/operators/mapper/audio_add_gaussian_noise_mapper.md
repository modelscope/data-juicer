# audio_add_gaussian_noise_mapper

Mapper to add Gaussian noise to audio samples.

This operator adds Gaussian noise to audio data with a specified probability. The
amplitude of the noise is randomly chosen between `min_amplitude` and `max_amplitude`.
If `save_dir` is provided, the modified audio files are saved in that directory;
otherwise, they are saved in the same directory as the input files. The `p` parameter
controls the probability of applying this transformation to each sample. If no audio is
present in the sample, it is returned unchanged.

Type 算子类型: **mapper**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_amplitude` | <class 'float'> | `0.001` | float unit: linear amplitude. |
| `max_amplitude` | <class 'float'> | `0.015` | float unit: linear amplitude. |
| `p` | <class 'float'> | `0.5` | float range: [0.0, 1.0].  Default: 0.5. |
| `save_dir` | <class 'str'> | `None` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/audio_add_gaussian_noise_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_audio_add_gaussian_noise_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)