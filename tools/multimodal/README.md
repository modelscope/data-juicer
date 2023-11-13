# Multimodal Tools

This folder contains some scripts and tools for multimodal datasets before and after using Data-Juicer.

## Dataset Format Conversion

Due to large format diversity among different multimodal datasets and works, 
Data-Juicer propose a novel intermediate format for multimodal dataset and 
provided several dataset format conversion tools for some popular multimodal 
works.

These tools consist of two types:
- Other format to Data-Juicer format: These tools are in `source_format_to_data_juicer_format` directory. They help to convert datasets in other formats to target datasets in Data-Juicer format.
- Data-Juicer format to other format: These tools are in `data_juicer_format_to_target_format` directory. They help to convert datasets in Data-Juicer formats to target datasets in target format.

For now, dataset formats that are supported by Data-Juicer are listed in the following table.

| Format     | source_format_to_data_juicer_format | data_juicer_format_to_target_format | Ref.                                                                                                             |
|------------|-------------------------------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------|
| LLaVA-like | `llava_to_dj.py`                    | `dj_to_llava.py`                    | [Format Description](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format) |

For all tools, you can run the following command to find out the usage of them:

```shell
# e.g. llava_to_dj.py
python tools/multimodal/source_format_to_data_juicer_format/llava_to_dj.py --help
```

Before using these tools, you might need to take a glance at the reference 
materials in the above tables for each format, to better know the detail format 
information and understand the arguments for each tool.

### Notice
There might be some tiny differences after converting a source dataset to Data-Juicer 
format and convert it back. However, these differences have nearly no effects 
on the semantics of datasets. Here we will show these tiny differences in detail 
for each source format.

#### LLaVA-like
The format of LLaVA-like datasets are defined [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md#dataset-format). 
Although it's simple, but in real scenarios, there might be some slight variations 
in some samples.

Here we take the [visual instruction tuning dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) as an example, 
and show how these variations influence the dataset format. The table below 
shows the number of different samples between the original dataset and the 
dataset after processing. There are 665,298 samples in the original dataset.

| process                                                                                | # of diff.  |
|----------------------------------------------------------------------------------------|-------------|
| 1. apply `llava_to_dj.py` and `dj_to_llava.py`                                         | 113,501     |
| 2. convert integer ids to string ids in the original dataset                           | 41,361      |
| 3. strip whitespaces before and after values of conversations in the original dataset  | 40,688      |
| 4. add `'model': ''` fields in the converted dataset                                   | 1           |

It's worth noticing that processes 2-4 won't influence the semantics of sample conversations in the dataset. 
Thus we think the dataset after conversion can align with the original dataset. 

Finally, the only 1 sample is different because there are some extra useless fields ("text", "markdown")
in the conversations, which is shown below. But the "from" and "value" fields are the same between original 
and converted datasets, so we can regard this sample is aligned with the original one as well.

```json
# original conversations
[
    {
        "from":"human",
        "value":"sentry self hosted api 504"
    },
    {
        "from":"gpt",
        "text":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\nSlow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n\nServer overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n\nLarge dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n\nAPI endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\nCheck your network connection and ensure that it is stable.\n\nVerify that the server hosting the API is not overloaded with requests.\n\nOptimize the query to retrieve only the necessary data.\n\nTry using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance.",
        "value":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n2. Verify that the server hosting the API is not overloaded with requests.\n3. Optimize the query to retrieve only the necessary data.\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance.",
        "markdown":{
            "type":"answer-markdown",
            "index":1,
            "answer":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n\n2. Verify that the server hosting the API is not overloaded with requests.\n\n3. Optimize the query to retrieve only the necessary data.\n\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance."
        }
    }
]

# converted conversations
[
    {
        "from":"human",
        "value":"sentry self hosted api 504"
    },
    {
        "from":"gpt",
        "value":"A 504 error when using the Sentry self-hosted API typically indicates that the API request timed out. This means that the server hosting the API did not receive a response from the client within the expected time frame.\n\nHere are a few possible reasons for a 504 error when using the Sentry self-hosted API:\n\n1. Slow network connection: The API request may be taking longer than expected due to a slow network connection. Check your network connection and try again.\n2. Server overload: The server hosting the API may be overloaded with requests. Try again later when the server load has decreased.\n3. Large dataset: If you are trying to retrieve a large amount of data, the API request may take longer than expected. Try breaking up the request into smaller chunks or optimizing the query to retrieve only the necessary data.\n4. API endpoint issues: There may be issues with the specific API endpoint you are using. Try using a different endpoint or contact the Sentry support team for assistance.\n\nTo resolve a 504 error when using the Sentry self-hosted API, try the following:\n\n1. Check your network connection and ensure that it is stable.\n2. Verify that the server hosting the API is not overloaded with requests.\n3. Optimize the query to retrieve only the necessary data.\n4. Try using a different API endpoint.\n\nIf the issue persists, contact the Sentry support team for assistance."
    }
]
```
