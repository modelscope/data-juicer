# Img-Diff: Contrastive Data Syhthesis for Multimodal Large Language Models


## Environment

```
transformers==4.36.2
```

For the other requirements, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main) and [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/).





## Image Pairs Generator

### step1 : generate caption pairs

```shell
# Img_Diff/pairs_generator/
$ bash gen.sh
```



### step2 : generate image pairs

```shell
# Img_Diff/pairs_generator/
$ bash gen_sdxl_new_data_ddp.sh
```





## Object Replacement Data Generator

### step1 : calculate image similarity

```shell
# Img_Diff/object_replacement/
$ bash cos_filter.sh
```



### step2 : image similarity filter

```shell
# Img_Diff/object_replacement/
$ python cos_count.py
```



### step3 : generate difference area

```shell
# Img_Diff/object_replacement/
$ bash generate_bbox.sh
```



### step4 : generate difference captions

```shell
# Img_Diff/object_replacement/
$ bash generate_final_data_new_edit.sh
```





## Object Removal Data Generator

```shell
# Img_Diff/object_removal/
$ bash run_generate_inpaint.sh
```

