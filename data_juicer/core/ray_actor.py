from functools import partial
import ray
import pyarrow

from data_juicer.ops.base_op import Filter, Mapper
from loguru import logger



def filter_batch(batch, filter_func):
    mask = pyarrow.array(filter_func(batch.to_pydict()))
    return batch.filter(mask)

@ray.remote(num_gpus=0.0) 
class Actor:
    def __init__(self, op, rank=None):

        self.op = op
        self._model_loaded = False  # 标记模型是否已加载
        self.rank = rank
        self.model = None
        self.processor = None
    
    def load_model(self):

        if self.op.use_cuda() and not self._model_loaded:
            
            self.model, self.processor = self.op.load_model(rank=self.rank)
            self._model_loaded = True

    def mapper_cuda(self, data):
        if not self._model_loaded:
            self.load_model()  # 确保调用前模型已加载
        data = self.op.process_single(data, self.model, self.processor)
        return data
    
    def mapper_cuda_batched(self, data):
        if not self._model_loaded:
            self.load_model()  # 确保调用前模型已加载
        data = self.op.process_batched_actor(data, self.model, self.processor)
        return data

    def mapper_cpu(self, data):
        # 处理数据
        processed_data = self.op.process_single(data)
        return processed_data
    
    def filter_cuda_single(self, data):
        if not self._model_loaded:
            self.load_model()
        data = self.op.compute_stats_single_actor(data, self.model, self.processor)
        keep = self.op.process_single(data)
        if keep:
            return data
        else:
            return None
        
    def filter_cuda_batched(self, data):
        if not self._model_loaded:
            self.load_model()
        # data = self.op.compute_stats_batched(data, self.model, self.processor)
        data = self.op.compute_stats_batched(data)
        keep_mask = list(self.op.process_batched(data))  # 将map对象转换为列表
    
        # 如果没有数据需要保留，返回None
        if not any(keep_mask):
            return None
        
        # 根据掩码过滤数据
        if isinstance(data, dict):
            # 如果data是字典（假设每个key对应一个列表）
            filtered_data = {
                key: [value for value, keep in zip(values, keep_mask) if keep]
                for key, values in data.items()
            }
        elif isinstance(data, list):
            # 如果data是列表
            filtered_data = [item for item, keep in zip(data, keep_mask) if keep]
        else:
            # 其他情况（如Ray Dataset的批处理）
            raise ValueError("Unsupported data type for batch filtering")
        
        return filtered_data


    def filter_cpu_single(self, data):
        data = self.op.compute_stats_single(data)
        keep = self.op.process_single(data)
        if keep:
            return data
        else:
            return None
        
    def filter_cpu_batched(self, data):
        # data = self.op.compute_stats_batched(data, self.model, self.processor)
        data = self.op.compute_stats_batched(data)
        keep_mask = list(self.op.process_batched(data))  # 将map对象转换为列表
    
        # 如果没有数据需要保留，返回None
        if not any(keep_mask):
            return None
        
        # 根据掩码过滤数据
        if isinstance(data, dict):
            # 如果data是字典（假设每个key对应一个列表）
            filtered_data = {
                key: [value for value, keep in zip(values, keep_mask) if keep]
                for key, values in data.items()
            }
        elif isinstance(data, list):
            # 如果data是列表
            filtered_data = [item for item, keep in zip(data, keep_mask) if keep]
        else:
            # 其他情况（如Ray Dataset的批处理）
            raise ValueError("Unsupported data type for batch filtering")
        
        return filtered_data

    
