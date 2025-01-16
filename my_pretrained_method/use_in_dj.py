# /tmp/zt_ori_mnt1/mnt/zt_pt_model/cogvlm2-video-llama3-chat
# /tmp/zt_ori_mnt1/mnt/zt_pt_model/llava-onevision-qwen2-7b-ov-hf

def cogvlm2_video_llama3_chat(video_path, query, model, tokenizer):
    import sys
    sys.path.append('/home/daoyuan_mm/data_juicer/data_juicer/my_pretrained_method/CogVLM2')
    from video_demo.cli_video_demo import load_video
    video = load_video(video_path, strategy='chat')
    inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=query,
            images=[video],
            history=[],
            template_version='chat'
        )
    inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
    gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": 128002,
            "top_k": 1,
            "do_sample": True,
            "top_p": 0.1,
            "temperature": 0.1,
        }
    with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)


def LLaVAOnevision(video_path,processor,model):
    import av
    from data_juicer.utils.mm_utils import read_video_pyav
    # Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos, up to 32 frames)
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    # For videos we have to feed a "video" type instead of "image"
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Why is this video funny?"},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to(device)

    out = model.generate(**inputs, max_new_tokens=60)
    response = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)