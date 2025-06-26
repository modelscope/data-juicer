import os
import base64
import requests
import unittest
from unittest.mock import patch, MagicMock

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.image_captioning_from_gpt4v_mapper import (
    ImageCaptioningFromGPT4VMapper,
)
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


@patch(
    "data_juicer.ops.mapper.image_captioning_from_gpt4v_mapper.call_gpt_vision_api",
    autospec=True,
)
class ImageCaptioningFromGPT4VMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")

    cat_path = os.path.join(data_path, "cat.jpg")
    img3_path = os.path.join(data_path, "img3.jpg")

    def _run_mapper(self, dataset: Dataset, op, caption_num=0):
        dataset = op.run(dataset)
        dataset_list = dataset.select_columns(column_names=["text"]).to_list()
        # assert the caption is generated successfully in terms of not_none
        # as the generated content is not deterministic
        self.assertEqual(len(dataset_list), caption_num)

    def test_no_eoc_special_token(self, mock_call_gpt_vision_api):

        ds_list = [
            {
                "text": f"{SpecialTokens.image}a photo of a cat",
                "images": [self.cat_path],
            },
            {
                "text": f"{SpecialTokens.image}a photo, a women with an umbrella",
                "images": [self.img3_path],
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningFromGPT4VMapper(
            api_key="mock_api_key", image_key="images", text_key="text"
        )
        mock_call_gpt_vision_api.return_value = "Mocked caption"
        self._run_mapper(dataset, op, caption_num=len(ds_list) * 2)

    def test_eoc_special_token(self, mock_call_gpt_vision_api):

        ds_list = [
            {
                "text": f"{SpecialTokens.image}a photo of a cat{SpecialTokens.eoc}",
                "images": [self.cat_path],
            },
            {
                "text": f"{SpecialTokens.image}a photo, a women with an umbrella{SpecialTokens.eoc}",  # noqa: E501
                "images": [self.img3_path],
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningFromGPT4VMapper(
            api_key="mock_api_key", image_key="images", text_key="text"
        )

        mock_call_gpt_vision_api.return_value = "Mocked caption"
        self._run_mapper(dataset, op, caption_num=len(ds_list) * 2)

    def test_keep_all(self, mock_call_gpt_vision_api):

        ds_list = [
            {
                "text": f"{SpecialTokens.image}a photo of a cat",
                "images": [self.cat_path],
            },
            {
                "text": f"{SpecialTokens.image}a photo, a women with an umbrella",
                "images": [self.img3_path],
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningFromGPT4VMapper(
            api_key="mock_api_key",
            image_key="images",
            text_key="text",
            any_or_all="all",
        )
        mock_call_gpt_vision_api.return_value = None
        self._run_mapper(dataset, op, caption_num=2)

    def test_custom_mode(self, mock_call_gpt_vision_api):

        ds_list = [
            {
                "text": f"{SpecialTokens.image}a photo of a cat",
                "images": [self.cat_path],
                "prompt": "test_prompt",
            },
            {
                "text": f"{SpecialTokens.image}a photo, a women with an umbrella",
                "images": [self.img3_path],
            },
        ]
        dataset = Dataset.from_list(ds_list)
        op = ImageCaptioningFromGPT4VMapper(
            api_key="mock_api_key",
            image_key="images",
            text_key="text",
            mode="custom",
            system_prompt="test_system_prompt",
            user_prompt="test_user_prompt",
            user_prompt_key="prompt",
        )
        mock_call_gpt_vision_api.return_value = "Mocked caption"
        self._run_mapper(dataset, op, caption_num=4)

    def test_error_mode(self, mock_call_gpt_vision_api):
        with self.assertRaises(ValueError):
            op = ImageCaptioningFromGPT4VMapper(mode="error")


@patch("requests.post", autospec=True)
class CallGPTVisionAPITest(DataJuicerTestCaseBase):

    def _call_gpt_vision_api(self):
        from data_juicer.ops.mapper.image_captioning_from_gpt4v_mapper import (
            call_gpt_vision_api,
        )

        return call_gpt_vision_api(
            "test_key",
            "system prompt",
            "user prompt",
            base64.b64encode(b"fake_image_data").decode("utf-8"),
        )

    def test_call_gpt_vision_api_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"text": "Generated text"}]}
        mock_post.return_value = mock_response

        result = self._call_gpt_vision_api()

        self.assertEqual(result, "Generated text")
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_no_choices(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response
        result = self._call_gpt_vision_api()
        self.assertIsNone(result)
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_request_error(self, mock_post):

        mock_post.side_effect = requests.exceptions.RequestException("Test error")

        result = self._call_gpt_vision_api()

        self.assertIsNone(result)
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_http_error_401(self, mock_post):

        http_error = requests.exceptions.HTTPError("Unauthorized")
        http_error.response = MagicMock(status_code=401)
        mock_post.side_effect = http_error
        result = self._call_gpt_vision_api()
        self.assertIsNone(result)
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_http_error_429(self, mock_post):

        http_error = requests.exceptions.HTTPError("Too Many Requests")
        http_error.response = MagicMock(status_code=429)
        mock_post.side_effect = http_error
        result = self._call_gpt_vision_api()
        self.assertIsNone(result)
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_connection_error(self, mock_post):

        mock_post.side_effect = requests.exceptions.ConnectionError()
        result = self._call_gpt_vision_api()
        self.assertIsNone(result)
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_timeout_error(self, mock_post):

        mock_post.side_effect = requests.exceptions.Timeout()
        result = self._call_gpt_vision_api()
        self.assertIsNone(result)
        mock_post.assert_called_once()

    def test_call_gpt_vision_api_other_error(self, mock_post):

        mock_post.side_effect = Exception("Other error")
        result = self._call_gpt_vision_api()
        self.assertIsNone(result)
        mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
