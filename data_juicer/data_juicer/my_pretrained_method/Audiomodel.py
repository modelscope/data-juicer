import sys
sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/SenseVoice')
from model import SenseVoiceSmall

model_dir = "/mnt1/daoyuan_mm/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir)


res = m.inference(
    data_in="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    language="zh", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
    **kwargs,
)

print(res)

