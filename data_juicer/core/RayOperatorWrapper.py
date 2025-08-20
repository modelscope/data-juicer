import ray

@ray.remote(num_gpus=0.0) 
class Actor:
    def __init__(self, op, rank=None):

        self.op = op
        self._model_loaded = False  # taggle to check if model is loaded
        self.rank = rank
        self.model = None
        self.processor = None
    
    def load_model(self):

        if self.op.use_cuda() and not self._model_loaded:
            
            self.model, self.processor = self.op.load_model(rank=self.rank)
            self._model_loaded = True

    def mapper_cuda(self, data):
        if not self._model_loaded:
            self.load_model()  # ensure model is loaded before processing
        # process data
        data = self.op.process_single_actor(data, self.model, self.processor)
        return data
    
    def mapper_cuda_batched(self, data):
        if not self._model_loaded:
            self.load_model()  # ensure model is loaded before processing
        # process data
        data = self.op.process_batched_actor(data, self.model, self.processor)
        return data

    def mapper_cpu(self, data):
        # process data
        processed_data = self.op.process_single(data)
        return processed_data
    
    def filter_cuda_single(self, data):
        if not self._model_loaded:
            self.load_model()
        # Call the Filter operator function
        data = self.op.compute_stats_single_actor(data, self.model, self.processor)
        keep = self.op.process_single(data)

        if keep:
            return data
        else:
            return None
        
    def filter_cuda_batched(self, data):
        if not self._model_loaded:
            self.load_model()
        data = self.op.compute_stats_batched(data, self.model, self.processor)
        # transform the map object to a list
        keep_mask = list(self.op.process_batched(data))  
    
        if not any(keep_mask):
            return None
        
        # filter data based on the keep_mask
        if isinstance(data, dict):
            filtered_data = {
                key: [value for value, keep in zip(values, keep_mask) if keep] for key, values in data.items()
            }
        elif isinstance(data, list):
            filtered_data = [item for item, keep in zip(data, keep_mask) if keep]
        else:
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
        data = self.op.compute_stats_batched(data)
        # transform the map object to a list
        keep_mask = list(self.op.process_batched(data))  

        if not any(keep_mask):
            return None

        # filter data based on the keep_mask
        if isinstance(data, dict):
            filtered_data = {
                key: [value for value, keep in zip(values, keep_mask) if keep] for key, values in data.items()
            }
        elif isinstance(data, list):

            filtered_data = [item for item, keep in zip(data, keep_mask) if keep]
        else:
            raise ValueError("Unsupported data type for batch filtering")
        
        return filtered_data

    
