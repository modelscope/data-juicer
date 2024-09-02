import io
from PIL import Image
import unittest
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.sdxl_prompt2prompt_mapper import SDXLPrompt2PromptMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SDXLPrompt2PromptMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'

    def _run_sdxl_prompt2prompt(self, enable_vllm=False):
        op = SDXLPrompt2PromptMapper(
            hf_diffusion='stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype="fp16"
        )


        ds_list = [{"caption1": "a chocolate cake",
                    "caption2": "a confetti apple cake"},
                    {"caption1": "a chocolate",
                    "caption2": "bread"}]

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        
        
        for temp_idx, sample in enumerate(dataset):
            for idx, img in enumerate(sample["images"]):
                img = Image.open(io.BytesIO(img["bytes"]))
                img.save(f"./test{str(temp_idx)}_{str(idx)}.jpg") 


    def test_sdxl_prompt2prompt(self):
        self._run_sdxl_prompt2prompt()



if __name__ == '__main__':
    unittest.main()