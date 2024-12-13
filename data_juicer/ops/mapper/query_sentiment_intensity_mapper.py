from typing import Dict, Optional

from data_juicer.utils.common_utils import nested_set
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper

OP_NAME = 'query_sentiment_intensity_mapper'


@OPERATORS.register_module(OP_NAME)
class QuerySentimentLabelMapper(Mapper):
    """
    Mapper to predict user's sentiment intensity label (-1 for 'negative',
    0 for 'neutral' and 1 for 'positive') in query. Input from query_key.
    Output intensity label and corresponding score for the query, which is
    store in 'sentiment.query_intensity' and
    'sentiment.query_intensity_score' in Data-Juicer meta field.
    """

    _accelerator = 'cuda'
    _batched_op = True

    DEFAULT_LABEL_TO_INTENSITY = {
        'negative': -1,
        'neutral': 0,
        'positive': 1,
    }

    def __init__(
            self,
            hf_model:
        str = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',  # noqa: E501 E131
            zh_to_en_hf_model: Optional[str] = 'Helsinki-NLP/opus-mt-zh-en',
            model_params: Dict = {},
            zh_to_en_model_params: Dict = {},
            *,
            label_to_intensity: Dict = None,
            **kwargs):
        """
        Initialization method.

        :param hf_model: Hugginface model ID to predict sentiment intensity.
        :param zh_to_en_hf_model: Translation model from Chinese to English.
            If not None, translate the query from Chinese to English.
        :param model_params: model param for hf_model.
        :param zh_to_en_model_params: model param for zh_to_hf_model.
        :param kwargs: Extra keyword arguments.
        :param label_to_intensity: Map the output labels to the intensities
            instead of the default mapper.
        """
        super().__init__(**kwargs)

        self.model_key = prepare_model(model_type='huggingface',
                                       pretrained_model_name_or_path=hf_model,
                                       return_pipe=True,
                                       pipe_task='text-classification',
                                       **model_params)

        if zh_to_en_hf_model is not None:
            self.zh_to_en_model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=zh_to_en_hf_model,
                return_pipe=True,
                pipe_task='translation',
                **zh_to_en_model_params)
        else:
            self.zh_to_en_model_key = None

        if label_to_intensity is not None:
            self.label_to_intensity = label_to_intensity
        else:
            self.label_to_intensity = self.DEFAULT_LABEL_TO_INTENSITY

    def process_batched(self, samples, rank=None):
        queries = samples[self.query_key]

        if self.zh_to_en_model_key is not None:
            translater, _ = get_model(self.zh_to_en_model_key, rank,
                                      self.use_cuda())
            results = translater(queries)
            queries = [item['translation_text'] for item in results]

        classifier, _ = get_model(self.model_key, rank, self.use_cuda())
        results = classifier(queries)
        intensities = [
            self.label_to_intensity[r['label']]
            if r['label'] in self.label_to_intensity else r['label']
            for r in results
        ]
        scores = [r['score'] for r in results]

        if Fields.meta not in samples:
            samples[Fields.meta] = [{} for val in intensities]
        for i in range(len(samples[Fields.meta])):
            samples[Fields.meta][i] = nested_set(
                samples[Fields.meta][i], MetaKeys.query_sentiment_intensity,
                intensities[i])
            samples[Fields.meta][i] = nested_set(
                samples[Fields.meta][i], MetaKeys.query_sentiment_score,
                scores[i])

        return samples
