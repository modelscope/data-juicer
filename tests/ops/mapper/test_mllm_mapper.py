import unittest
from data_juicer.ops.mapper.mllm_mapper import MllmMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class MllmMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'
    image_key = "images"

    def _run_mllm(self, enable_vllm=False):
        op = MllmMapper(
            hf_model='liuhaotian/llava-v1.6-vicuna-7b',
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=512
        )

        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
        img2_path = os.path.join(data_path, 'img2.jpg')
        img3_path = os.path.join(data_path, 'img3.jpg')
      
        samples = [
            {self.text_key: 'Describe this image.', self.image_key: [img2_path, img3_path]},
        ]

        for sample in samples:
            result = op.process(sample)
            print(f'Output results: {result}')

    def test_mllm(self):
        self._run_mllm()


if __name__ == '__main__':
    unittest.main()
