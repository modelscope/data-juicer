from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

TEXT_EMBEDDING_MAPPER = "text_embedding_mapper"


@OPERATORS.register_module(TEXT_EMBEDDING_MAPPER)
class TextEmbeddingMapper(Mapper):
    """Mapper to compute text embeddings for each sample."""

    _accelerator = "cuda"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        embedding_field: str = "embedding",
        *args,
        **kwargs,
    ):
        """
        Initialization method.
        :param model_name: The name of the sentence-transformer model to use.
        :param embedding_field: The field name to store the computed embedding.
        :param args: Extra args.
        :param kwargs: Extra args.
        """
        # Set memory requirements based on typical embedding model sizes
        kwargs.setdefault("mem_required", "4GB")
        super().__init__(*args, **kwargs)
        self.embedding_field = embedding_field

        # Use DJ's model management utility
        self.model_key = prepare_model(model_type="sentence_transformer", model_name=model_name)

    def process_single(self, sample, rank=None):
        # Skip if embedding is already computed
        if self.embedding_field in sample:
            return sample

        # Ensure there is text to process
        if self.text_key not in sample or not sample[self.text_key]:
            sample[self.embedding_field] = None  # or np.array([])
            return sample

        # Get the model for the current process/rank
        model = get_model(self.model_key, rank, self.use_cuda())

        # Compute embedding
        # The model from sentence-transformers can take a single string
        # or a list of strings. Here we process one by one.
        # For batched OP, this logic would be in `process_batched`.
        embedding = model.encode(
            sample[self.text_key], device=model.device, show_progress_bar=False
        )  # No progress bar for single sample

        sample[self.embedding_field] = embedding.tolist()  # Store as list for JSON compatibility
        return sample
