# DJ_服务化
中文 | [English Page](DJ_service.md) 

为了进一步提升Data-Juicer用户体验，我们新增了基于 API 的服务功能（Service API）和 MCP 服务，使用户能够以更便捷的方式集成和使用 Data-Juicer 的强大算子池。通过服务功能，用户无需深入了解框架的底层实现细节，即可快速构建数据处理流水线，并与现有系统无缝对接。用户也可通过该服务实现不同project之间的环境隔离。本文档将详细介绍如何启动和使用这两种服务功能，帮助您快速上手并充分发挥 Data-Juicer 的潜力。

## API服务化
### 启动服务
执行如下代码：
```bash
uvicorn service:app
```

### API调用
API支持调用Data-Juicer所有`__init__.py`中的函数和类（调用类的某个函数）。函数调用GET调用，类通过POST调用。

#### 协议

#### url路径
采用GET调用函数，url路径与Data-Juicer库中函数引用路径一致，如`from data_juicer.config import init_configs`对应路径为`data_juicer/config/init_configs`。采用POST调用类的某个函数，url路径与Data-Juicer库中类路径拼接上函数名，如调用`TextLengthFIlter`算子的`compute_stats_batched`函数，对应路径为`data_juicer/ops/filter/TextLengthFilter/compute_stats_batched`。

##### 参数
进行GET和POST调用时，会自动将参数转化为list，同时，查询参数不支持字典传输。于是，如果传递的参数的value是list或dict，我们统一用`json.dumps`传输，并在前面添加特殊符号`<json_dumps>`与一般的string进行区分。

##### 特例
1. 针对`cfg`参数，我们默认采用`json.dumps`传输，无需添加特殊符号`<json_dumps>`。
2. 针对`dataset`参数，允许用户传输`dataset`在服务器上的路径，sever将加载dataset。
3. 允许用户设定`skip_return`参数，为`True`时将不返回函数调用的结果，避免一些网络无法传输带来的错误。

#### 函数调用
采用GET调用，url路径与Data-Juicer库中函数引用路径一致，查询参数用来传递函数的参数。

例如，可用如下python代码调用Data-Juicer的参数init函数`init_configs`，获取Data-Juicer所有参数。

```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/config/init_configs'
params = {"args": json_prefix + json.dumps(['--config', './configs/demo/process.yaml'])}
response = requests.get(url, params=params)
print(json.loads(response.text))
```

对应的curl代码如下：

```bash
curl -G "http://localhost:8000/data_juicer/config/init_configs" \
     --data-urlencode "args=--config" \
     --data-urlencode "args=./configs/demo/process.yaml"
```

#### 类的函数调用
采用POST调用，url路径与Data-Juicer库中类路径拼接上函数名，查询参数用来传递函数的参数，JSON字段用来传递类构造函数所需参数。

例如，可用如下python代码调用Data-Juicer的`TextLengthFIlter`算子。
```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched'
params = {'samples': json_prefix + json.dumps({'text': ['12345', '123'], '__dj__stats__': [{}, {}]})}
init_json = {'min_len': 4, 'max_len': 10}
response = requests.post(url, params=params, json=init_json)
print(json.loads(response.text))
```

对应的curl代码如下：

```bash
curl -X POST \
  "http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched?samples=%3Cjson_dumps%3E%7B%22text%22%3A%20%5B%2212345%22%2C%20%22123%22%5D%2C%20%22__dj__stats__%22%3A%20%5B%7B%7D%2C%20%7B%7D%5D%7D" \
  -H "Content-Type: application/json" \
  -d '{"min_len": 4, "max_len": 10}'
```


注：如果需要调用`Executor`类或`Analyzer`类的`run`函数进行数据处理和数据分析，需要先调用`init_configs`或`get_init_configs`函数获取完整的Data-Juicer参数来构造这两个类。具体可参考如下演示。

### 演示
我们结合[AgentScope](https://github.com/modelscope/agentscope)实现了用户通过自然语言调用Data-Juicer算子进行数据清洗的功能，算子采用API服务的方式进行调用。具体代码请参考[这里](../demos/api_service)。

## MCP服务器

### 概览

Data-Juicer MCP 服务器提供数据处理算子，以协助完成数据清洗、过滤、去重等任务。
为了适应不同的使用场景，我们提供两种服务器供选用：

Recipe-Flow: 允许根据算子的类型和标签进行筛选，并支持将多个算子组合成一个数据菜谱来运行。
Granular-Operators: 将每个算子作为一个独立的工具提供，您可以灵活地通过环境变量指定需要使用的算子列表，从而构建定制化的数据处理管道。

请注意，Data-Juicer MCP 服务器目前处于早期开发阶段。其功能和可用工具可能会随着我们继续开发和改进服务器而发生变化和扩展。

### Recipe-Flow

1. `get_data_processing_ops`
   - 根据指定的类型和标签检索可用的数据处理算子列表（若不指定，则返回全部算子）
   - 输入：
     - `op_type` (str, optional): 要检索的数据处理算子类型
     - `tags` (List[str], optional): 用于过滤算子的标签列表
     - `match_all` (bool): 是否所有指定的标签都必须匹配。默认为 True
   - 返回：包含可用算子详细信息的字典

2. `run_data_recipe`
   - 执行数据菜谱
   - 输入：
     - `dataset_path` (str): 要处理的数据集路径
     - `process` (List[Dict]): 要执行的处理步骤列表，字典包含算子名称和参数字典
     - `export_path` (str, optional): 导出数据集的路径，默认为 None，这意味着数据集将导出到 './outputs'
   - 返回：执行结果的字符串

针对特定数据处理请求，MCP client 应先调用`get_data_processing_ops`获取相关的算子信息，从中选择匹配需求的算子，然后调用`run_data_recipe`运行选择的算子组合。

#### 配置

##### 使用 Claude Desktop

将以下内容添加到您的 `claude_desktop_config.json`：

```json
"mcpServers": {
  "DJ_search_op": {
    "command": "/absolute/path/to/python",
    "args": [
      "/absolute/path/to/data_juicer/tools/DJ_mcp_recipe_flow.py"
    ]
  }
}
```
### Granular-Operators

默认情况下，该 MCP 服务器将返回所有Data-Juicer算子工具，每个工具都独立运行。

可通过指定环境变量 `DJ_OPS_LIST_PATH` 控制 MCP 服务器返回的算子工具：
1. 创建一个`.txt`文件
2. 将算子名称添加到文件中，例如：[dj_test.txt](../data_juicer/tools/dj_test.txt)
3. 将算子列表路径设置为环境变量 `DJ_OPS_LIST_PATH`

#### 配置

##### 使用 Claude Desktop

将以下内容添加到您的 `claude_desktop_config.json`：

```json
"mcpServers": {
  "DJ_search_op": {
    "command": "/absolute/path/to/python",
    "args": [
      "/absolute/path/to/data_juicer/tools/DJ_mcp_granular_ops.py"
    ],
    "env": {
      "DJ_OPS_LIST_PATH": "/absolute/path/to/ops_list.txt"
    }
  }
}
```


### 查找你的 Python 路径
要查找 Python 可执行文件路径，请使用以下命令：

Windows (Command Prompt/Terminal):
```
where python
```
Linux/macOS (Terminal):
```
which python
```
