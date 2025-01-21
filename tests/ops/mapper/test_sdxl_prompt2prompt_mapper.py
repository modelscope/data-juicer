import io
from PIL import Image
import unittest
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.sdxl_prompt2prompt_mapper import SDXLPrompt2PromptMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SDXLPrompt2PromptMapperTest(DataJuicerTestCaseBase):

    hf_diffusion = 'stabilityai/stable-diffusion-xl-base-1.0'

    text_key = 'text'

    text_key_second = "caption1"
    text_key_third = "caption2"

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_diffusion)

    def _run_sdxl_prompt2prompt(self):
        op = SDXLPrompt2PromptMapper(
            hf_diffusion=self.hf_diffusion,
            torch_dtype="fp16",
            text_key_second=self.text_key_second,
            text_key_third=self.text_key_third
        )

        ds_list = [{self.text_key_second: "a chocolate cake",
                    self.text_key_third: "a confetti apple bread"},
                   {self.text_key_second: "a chocolate",
                    self.text_key_third: "bread"}]

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        
        for temp_idx, sample in enumerate(dataset):
            self.assertIn(op.image_key, sample)
            self.assertGreater(len(sample[op.image_key]), 0)
            for idx, img in enumerate(sample["images"]):
                img = Image.open(io.BytesIO(img["bytes"]))
                img.save(f"./test{str(temp_idx)}_{str(idx)}.jpg") 

    def test_sdxl_prompt2prompt(self):
        self._run_sdxl_prompt2prompt()


if __name__ == '__main__':
    unittest.main()