import faiss
import numpy as np
from loguru import logger

from ..base_op import OPERATORS, StatefulOP

EMBEDDING_INDEXER = "embedding_indexer"


@OPERATORS.register_module(EMBEDDING_INDEXER)
class EmbeddingIndexer(StatefulOP):
    """
    StatefulOP to aggregate embeddings and build a FAISS index in the
    global context.
    """

    def __init__(self, embedding_field: str = "embedding", index_context_key: str = "faiss_index", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_field = embedding_field
        self.index_context_key = index_context_key

    def process(self, dataset, context: dict):
        if self.context_key in context:
            logger.info(f"Context key [{self.context_key}] already exists. Skipping aggregation.")
            return dataset

        logger.info(f"Aggregating embeddings from field [{self.embedding_field}]...")

        # Efficiently get the whole column. This is much faster than iterating.
        all_embeddings = dataset.select_columns(self.embedding_field)[self.embedding_field]

        # Filter out any None values if some samples had no text
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]

        embeddings_np = np.array(valid_embeddings).astype("float32")
        dimension = embeddings_np.shape[1]

        # Create a simple FAISS index
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product similarity
        index.add(embeddings_np)

        context[self.index_context_key] = index
        logger.info(f"Built FAISS index with {index.ntotal} vectors and stored in context.")

        return dataset
