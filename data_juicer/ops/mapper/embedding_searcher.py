import numpy as np

from ..base_op import OPERATORS, Mapper

EMBEDDING_SEARCHER = "embedding_searcher"


@OPERATORS.register_module(EMBEDDING_SEARCHER)
class EmbeddingSearcher(Mapper):
    """
    Mapper to find top_k similar samples using a FAISS index from the
    global context.
    """

    def __init__(
        self,
        top_k: int = 10,
        embedding_field: str = "embedding",
        index_context_key: str = "faiss_index",
        search_result_field: str = "similar_samples",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.embedding_field = embedding_field
        self.index_context_key = index_context_key
        self.search_result_field = search_result_field

    def process_single(self, sample):
        if self.index_context_key not in self.context or self.embedding_field not in sample:
            sample[self.search_result_field] = None
            return sample

        index = self.context.get(self.index_context_key)
        if not index or index.ntotal == 0:
            sample[self.search_result_field] = None
            return sample

        query_embedding = np.array(sample[self.embedding_field]).astype("float32").reshape(1, -1)

        # Search the index
        # We search for k+1 because the query itself will be the top result
        distances, indices = index.search(query_embedding, self.top_k + 1)

        # Remove the query itself from the results
        results = []
        for i in range(1, len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx != -1:  # FAISS uses -1 for no result
                results.append({"id": int(idx), "similarity": float(dist)})

        sample[self.search_result_field] = results
        return sample
