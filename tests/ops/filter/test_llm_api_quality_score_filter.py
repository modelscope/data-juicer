import unittest

from loguru import logger

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.llm_api_quality_score_filter import LLMAPIQualityScoreFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class LLMAPIQualityScoreFilterTest(DataJuicerTestCaseBase):
    # before running this test, set below environment variables:
    # export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
    # export OPENAI_API_KEY=your_dashscope_key
    api_model = 'qwen2.5-72b-instruct'

    def _run_test(self, dataset: Dataset, target_list, op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                         column=[{}] * dataset.num_rows)

        dataset = dataset.map(
            op.compute_stats,
            batch_size=op.batch_size
            )
        logger.info(dataset.to_list())
        dataset = dataset.filter(op.process, batch_size=op.batch_size)
        dataset_test = dataset.select_columns(column_names=['text'])
        res_list = dataset_test.to_list()
        self.assertEqual(res_list, target_list)
        return dataset

    def test_default_case(self):

        ds_list = [{
            'text':
            "The cat fly in sky and eat stars. It very happy because stars is tasty."
        }, {
            'text':
            "Cats are known for their agility and independence. They often sleep during the day and are active at night. Some people believe cats bring good luck."
        }, {
            'text':
            "Cats are domesticated animals known for their agility, intelligence, and independent nature. Research shows that they spend approximately 70% of their lives sleeping, which helps conserve energy for hunting. Unlike dogs, cats are obligate carnivores, meaning their diet must consist primarily of meat to meet nutritional needs."
        }]
        tgt_list = [{
            'text':
            "Cats are known for their agility and independence. They often sleep during the day and are active at night. Some people believe cats bring good luck."
        }, {
            'text':
            "Cats are domesticated animals known for their agility, intelligence, and independent nature. Research shows that they spend approximately 70% of their lives sleeping, which helps conserve energy for hunting. Unlike dogs, cats are obligate carnivores, meaning their diet must consist primarily of meat to meet nutritional needs."
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMAPIQualityScoreFilter(api_model=self.api_model)
        dataset= self._run_test(dataset, tgt_list, op)

    def test_rft_data(self):
        ds_list = [{
            "text": "What is the fastest animal?",
            "analysis": "The fastest animal is fish because they swim very fast in water.",
            "answer": "Fish."
        }, {
            "text": "Why do leaves change color in autumn?",
            "analysis": "Leaves change color because of the decrease in sunlight and temperature. Chlorophyll breaks down, revealing other pigments like yellow and orange.",
            "answer": "Due to less sunlight and colder temperatures, chlorophyll breaks down, showing other colors."
        }, {
            "text": "How does photosynthesis work?",
            "analysis": "Photosynthesis is the process by which plants convert light energy into chemical energy. Chlorophyll absorbs sunlight, which drives the conversion of carbon dioxide and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells and is essential for life on Earth.",
            "answer": "Plants use chlorophyll to absorb sunlight, converting carbon dioxide and water into glucose and oxygen through a process that occurs in chloroplasts."
        }]
        tgt_list = [{
            "text": "Why do leaves change color in autumn?"
        }, {
            "text": "How does photosynthesis work?"
        }]
        dataset = Dataset.from_list(ds_list)
        op = LLMAPIQualityScoreFilter(
            api_model=self.api_model,
            input_keys=['text', 'analysis', 'answer'],
            field_names=['Query', 'Analysis', 'Answer'],
        )
        dataset= self._run_test(dataset, tgt_list, op)

if __name__ == '__main__':
    unittest.main()
