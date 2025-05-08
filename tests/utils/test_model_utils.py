import unittest
from unittest.mock import patch, MagicMock

from data_juicer.utils.model_utils import (
    get_backup_model_link,
    prepare_simple_aesthetics_model,
    prepare_api_model,
    prepare_huggingface_model,
    prepare_vllm_model,
    prepare_diffusion_model,
    prepare_fasttext_model,
    prepare_kenlm_model,
    prepare_nltk_model,
    prepare_sentencepiece_model,
    prepare_video_blip_model,
    prepare_fastsam_model,
    prepare_sdxl_prompt2prompt,
    prepare_model,
    get_model,
    free_models,
    prepare_recognizeAnything_model,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# other funcs are called by ops already
class ModelUtilsTest(DataJuicerTestCaseBase):

    def test_get_backup_model_link(self):
        test_data = [
            ('lid.176.bin', 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/'),  # exact match
            ('zh.sp.model', 'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/'),  # pattern match
            ('invalid_model_name', None),  # invalid model name
        ]
        for model_name, expected_link in test_data:
            self.assertEqual(get_backup_model_link(model_name), expected_link)

    @patch('data_juicer.utils.model_utils.aes_pred')
    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_simple_aesthetics_model(self, mock_transformers, mock_aes_pred):
        # Test V1 model
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_aes_pred.AestheticsPredictorV1.from_pretrained.return_value = mock_model
        mock_transformers.CLIPProcessor.from_pretrained.return_value = mock_processor

        model = prepare_simple_aesthetics_model('v1_model')
        self.assertEqual(model[0], mock_model)
        self.assertEqual(model[1], mock_processor)
        mock_aes_pred.AestheticsPredictorV1.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()

        # Test V2 Linear model
        mock_aes_pred.reset_mock()
        mock_transformers.reset_mock()
        mock_aes_pred.AestheticsPredictorV2Linear.from_pretrained.return_value = mock_model
        model = prepare_simple_aesthetics_model('v2_linear_model')
        self.assertEqual(model[0], mock_model)
        self.assertEqual(model[1], mock_processor)
        mock_aes_pred.AestheticsPredictorV2Linear.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()

        # Test V2 ReLU model
        mock_aes_pred.reset_mock()
        mock_transformers.reset_mock()
        mock_aes_pred.AestheticsPredictorV2ReLU.from_pretrained.return_value = mock_model
        model = prepare_simple_aesthetics_model('v2_relu_model')
        self.assertEqual(model[0], mock_model)
        self.assertEqual(model[1], mock_processor)
        mock_aes_pred.AestheticsPredictorV2ReLU.from_pretrained.assert_called_once()
        mock_transformers.CLIPProcessor.from_pretrained.assert_called_once()

        # Test invalid model
        with self.assertRaises(ValueError):
            prepare_simple_aesthetics_model('invalid_model')

    @patch('data_juicer.utils.model_utils.openai')
    @patch('data_juicer.utils.model_utils.tiktoken')
    def test_prepare_api_model(self, mock_tiktoken, mock_openai):
        # Test basic API model
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_processor = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_processor

        model = prepare_api_model('test_model')
        self.assertEqual(model._client, mock_client)

        # Test with processor
        mock_openai.OpenAI.reset_mock()
        mock_tiktoken.encoding_for_model.reset_mock()
        model, processor = prepare_api_model('test_model', return_processor=True)
        self.assertEqual(model._client, mock_client)
        self.assertEqual(processor, mock_processor)
        mock_tiktoken.encoding_for_model.assert_called_once()

    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_huggingface_model(self, mock_transformers):
        # Test model with processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model, processor = prepare_huggingface_model('test_model', return_model=True)
        self.assertEqual(model, mock_model)
        self.assertEqual(processor, mock_processor)

        # Test processor only
        processor = prepare_huggingface_model('test_model', return_model=False)
        self.assertEqual(processor, mock_processor)

    @patch('data_juicer.utils.model_utils.os')
    def test_prepare_vllm_model(self, mock_os):
        # Create a mock vllm module
        mock_vllm = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_vllm.LLM.return_value = mock_model
        mock_model.get_tokenizer.return_value = mock_tokenizer

        # Replace the vllm module in model_utils
        import data_juicer.utils.model_utils as model_utils
        model_utils.vllm = mock_vllm

        # Test basic functionality
        model, tokenizer = prepare_vllm_model('test_model')
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)
        mock_vllm.LLM.assert_called_once()
        mock_model.get_tokenizer.assert_called_once()

        # Test environment setup
        mock_os.environ.__setitem__.assert_called_with('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

        # Test device handling
        mock_vllm.LLM.reset_mock()
        model, _ = prepare_vllm_model('test_model', device='cuda:0')
        mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto', device='cuda')

        # Test model parameters
        mock_vllm.LLM.reset_mock()
        model_params = {'tensor_parallel_size': 2, 'max_model_len': 2048}
        model, _ = prepare_vllm_model('test_model', **model_params)
        mock_vllm.LLM.assert_called_once_with(model='test_model', generation_config='auto', **model_params)

    @patch('data_juicer.utils.model_utils.diffusers')
    def test_prepare_diffusion_model(self, mock_diffusers):
        mock_model = MagicMock()
        mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value = mock_model

        model = prepare_diffusion_model('test_model', 'text2image')
        self.assertEqual(model, mock_model)

        # Test invalid diffusion type
        with self.assertRaises(ValueError):
            prepare_diffusion_model('test_model', 'invalid_type')

    @patch('data_juicer.utils.model_utils.fasttext')
    def test_prepare_fasttext_model(self, mock_fasttext):
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model

        model = prepare_fasttext_model('test_model')
        self.assertEqual(model, mock_model)

    @patch('data_juicer.utils.model_utils.kenlm')
    def test_prepare_kenlm_model(self, mock_kenlm):
        mock_model = MagicMock()
        mock_kenlm.Model.return_value = mock_model

        model = prepare_kenlm_model('en')
        self.assertEqual(model, mock_model)

    @patch('data_juicer.utils.model_utils.nltk')
    def test_prepare_nltk_model(self, mock_nltk):
        mock_model = MagicMock()
        mock_nltk.data.load.return_value = mock_model

        model = prepare_nltk_model('en')
        self.assertEqual(model, mock_model)

        # Test invalid language
        with self.assertRaises(AssertionError):
            prepare_nltk_model('invalid_lang')

    @patch('data_juicer.utils.model_utils.sentencepiece')
    def test_prepare_sentencepiece_model(self, mock_sentencepiece):
        mock_model = MagicMock()
        mock_sentencepiece.SentencePieceProcessor.return_value = mock_model

        model = prepare_sentencepiece_model('test_model')
        self.assertEqual(model, mock_model)

    @patch('data_juicer.utils.model_utils.transformers')
    def test_prepare_video_blip_model(self, mock_transformers):
        # Set up mock classes and methods
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Set up the mock transformers module
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.Blip2ForConditionalGeneration = MagicMock()
        mock_transformers.Blip2ForConditionalGeneration.from_pretrained.return_value = mock_model
        mock_transformers.Blip2VisionModel = MagicMock()
        
        # Mock the custom VideoBlipForConditionalGeneration class
        class MockVideoBlipForConditionalGeneration:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        mock_transformers.Blip2ForConditionalGeneration = MockVideoBlipForConditionalGeneration

        model, processor = prepare_video_blip_model('test_model')
        self.assertEqual(model, mock_model)
        self.assertEqual(processor, mock_processor)

    @patch('data_juicer.utils.model_utils.ultralytics')
    def test_prepare_fastsam_model(self, mock_ultralytics):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Make .to() return the same mock
        mock_ultralytics.FastSAM.return_value = mock_model

        model = prepare_fastsam_model('test_model')
        self.assertEqual(model, mock_model)
        mock_model.to.assert_called_once()  # Verify .to() was called

    @patch('data_juicer.utils.model_utils.diffusers')
    def test_prepare_sdxl_prompt2prompt(self, mock_diffusers):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Make .to() return the same mock
        mock_diffusers.AutoPipelineForText2Image.from_pretrained.return_value = mock_model

        model = prepare_sdxl_prompt2prompt('test_model', mock_diffusers.AutoPipelineForText2Image)
        self.assertEqual(model, mock_model)
        mock_model.to.assert_called_once()  # Verify .to() was called

    def test_prepare_model(self):
        # Test valid model type
        model_func = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        self.assertIsNotNone(model_func)

        # Test invalid model type
        with self.assertRaises(AssertionError):
            prepare_model('invalid_type')

    @patch('data_juicer.utils.model_utils.transformers')
    def test_get_model(self, mock_transformers):
        # Test getting a model
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model_key = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        model = get_model(model_key)
        self.assertIsNotNone(model)

        # Test getting a model with CUDA
        model = get_model(model_key, use_cuda=True)
        self.assertIsNotNone(model)

    @patch('data_juicer.utils.model_utils.transformers')
    def test_free_models(self, mock_transformers):
        # Test freeing models
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        model_key = prepare_model('huggingface', pretrained_model_name_or_path='test_model')
        get_model(model_key)
        free_models()
        # No assertion needed, just checking it doesn't raise an exception

    @patch('data_juicer.utils.model_utils.ram')
    @patch('data_juicer.utils.model_utils.check_model')
    def test_prepare_recognizeAnything_model(self, mock_check_model, mock_ram):
        # Set up mocks
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_ram.models.ram_plus.return_value = mock_model
        mock_check_model.return_value = 'test_model_path'

        # Test normal initialization
        model = prepare_recognizeAnything_model(
            pretrained_model_name_or_path='test_model.pth',
            input_size=384
        )
        self.assertEqual(model, mock_model)
        mock_ram.models.ram_plus.assert_called_once_with(
            pretrained='test_model_path',
            image_size=384,
            vit='swin_l'
        )
        mock_model.to.assert_called_once_with('cpu')
        mock_model.eval.assert_called_once()

        # Reset mocks for next test
        mock_model.reset_mock()
        mock_ram.models.ram_plus.reset_mock()
        mock_check_model.reset_mock()
        mock_check_model.return_value = 'test_model_path'

        # Test with custom device
        model = prepare_recognizeAnything_model(
            pretrained_model_name_or_path='test_model.pth',
            input_size=384,
            device='cuda:0'
        )
        self.assertEqual(model, mock_model)
        mock_model.to.assert_called_once_with('cuda:0')

        # Reset mocks for error handling test
        mock_model.reset_mock()
        mock_ram.models.ram_plus.reset_mock()
        mock_check_model.reset_mock()
        mock_check_model.side_effect = ['test_model_path', 'test_model_path_force']
        mock_ram.models.ram_plus.side_effect = [RuntimeError(), mock_model]

        # Test error handling with force=True
        model = prepare_recognizeAnything_model(
            pretrained_model_name_or_path='test_model.pth',
            input_size=384
        )
        self.assertEqual(model, mock_model)
        self.assertEqual(mock_check_model.call_count, 2)
        self.assertEqual(mock_ram.models.ram_plus.call_count, 2)

    @patch('data_juicer.utils.lazy_loader.LazyLoader._install_github_deps')
    @patch('data_juicer.utils.lazy_loader.importlib.import_module')
    @patch('data_juicer.utils.model_utils.LazyLoader')
    def test_ram_lazy_loader(self, mock_lazy_loader_class, mock_import, mock_install_github):
        # Create a mock LazyLoader instance
        mock_lazy_loader = MagicMock()
        mock_lazy_loader._module_name = 'ram'
        mock_lazy_loader._package_url = 'git+https://github.com/xinyu1205/recognize-anything.git'
        mock_lazy_loader_class.return_value = mock_lazy_loader
        
        # Create a mock module with models attribute
        mock_module = MagicMock()
        mock_module.models = MagicMock()
        
        # Mock the import to fail first time to trigger GitHub installation
        mock_import.side_effect = [ImportError("Module not found"), mock_module]
        
        # Import ram after setting up mocks
        from data_juicer.utils.model_utils import ram
        
        # Access the module to trigger loading
        _ = ram.models
        
        # Verify installation was attempted
        mock_install_github.assert_called_once_with(
            'git+https://github.com/xinyu1205/recognize-anything.git',
            use_uv=True
        )
        self.assertEqual(mock_import.call_count, 2)  # Called twice: once for import, once after installation


if __name__ == '__main__':
    unittest.main()
