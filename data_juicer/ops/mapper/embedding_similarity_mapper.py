import numpy as np

from ..base_op import OPERATORS, Mapper

EMBEDDING_SIMILARITY_MAPPER = "embedding_similarity_mapper"


@OPERATORS.register_module(EMBEDDING_SIMILARITY_MAPPER)
class EmbeddingSimilarityMapper(Mapper):
    """
    Mapper to compute the similarity of each sample's embedding against all
    other embeddings stored in the global context.
    """

    def __init__(
        self,
        embedding_field: str = "embedding",
        context_key: str = "all_embeddings",
        similarity_field: str = "similarities",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_field = embedding_field
        self.context_key = context_key
        self.similarity_field = similarity_field

    def process_single(self, sample):
        # Check if context is ready and sample has an embedding
        if (
            self.context_key not in self.context
            or self.embedding_field not in sample
            or sample[self.embedding_field] is None
        ):
            sample[self.similarity_field] = None
            return sample

        all_embeddings = self.context.get(self.context_key)
        if all_embeddings.size == 0:
            sample[self.similarity_field] = []
            return sample

        current_embedding = np.array(sample[self.embedding_field])

        # Compute dot product (inner product)
        similarities = np.dot(all_embeddings, current_embedding.T)

        sample[self.similarity_field] = similarities.tolist()
        return sample
