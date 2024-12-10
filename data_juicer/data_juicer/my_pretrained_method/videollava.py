import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from PIL import Image
import requests
import numpy as np
import av
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def videollava_generate(model, processor, prompt="USER: <video>Describe the video. ASSISTANT:", video_path = '/home/daoyuan_mm/1WBgdccmCU4_57__0.mp4'):
    container = av.open(video_path)

    # sample uniformly 8 frames from the video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)

    inputs = processor(text=prompt, videos=clip, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate
    generate_ids = model.generate(**inputs, max_length=200)
    # print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
    out = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    index = out.find('ASSISTANT: ')
    if index != -1:
        return out[index + len('ASSISTANT: '):]
    else:
        return ""


# model = VideoLlavaForConditionalGeneration.from_pretrained("/mnt1/daoyuan_mm/Video-LLaVA-7B-hf")
# processor = VideoLlavaProcessor.from_pretrained("/mnt1/daoyuan_mm/Video-LLaVA-7B-hf")
# model = model.cuda()

# instruct = "Is the person a child?"
    
# prompt = 'USER: <video>' + instruct + '  ASSISTANT:'
# print(videollava_generate(model, processor,prompt))