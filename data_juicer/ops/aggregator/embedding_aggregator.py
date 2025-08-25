import numpy as np
from loguru import logger

from ..base_op import OPERATORS, StatefulOP

EMBEDDING_AGGREGATOR = "embedding_aggregator"


@OPERATORS.register_module(EMBEDDING_AGGREGATOR)
class EmbeddingAggregator(StatefulOP):
    """
    StatefulOP to aggregate all embeddings from the dataset into the
    global context.
    """

    def __init__(self, embedding_field: str = "embedding", context_key: str = "all_embeddings", *args, **kwargs):
        """
        :param embedding_field: Field name of the embedding in each sample.
        :param context_key: Key under which to store the aggregated embeddings
                            in the global context.
        """
        super().__init__(*args, **kwargs)
        self.embedding_field = embedding_field
        self.context_key = context_key

    def process(self, dataset, context: dict):
        """
        Iterates through the dataset, collects all embeddings, and stores
        them as a NumPy array in the shared context.
        """
        if self.context_key in context:
            logger.info(f"Context key [{self.context_key}] already exists. Skipping aggregation.")
            return dataset

        logger.info(f"Aggregating embeddings from field [{self.embedding_field}]...")

        # Efficiently get the whole column. This is much faster than iterating.
        all_embeddings = dataset.select_columns(self.embedding_field)[self.embedding_field]

        # Filter out any None values if some samples had no text
        valid_embeddings = [emb for emb in all_embeddings if emb is not None]

        if not valid_embeddings:
            logger.warning("No valid embeddings found to aggregate.")
            context[self.context_key] = np.array([])
            return dataset

        context[self.context_key] = np.array(valid_embeddings)
        logger.info(f"Aggregated {len(valid_embeddings)} embeddings into context key [{self.context_key}].")

        return dataset
