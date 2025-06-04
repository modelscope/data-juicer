import os
import tempfile
import unittest
import json
import pandas as pd
from tools.parquet2jsonl import parquet_to_jsonl, get_parquet_file_names, convert_parquet_to_jsonl


class TestParquetToJsonl(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.parquet_file = os.path.join(self.temp_dir.name, 'test.parquet')
        self.output_dir = os.path.join(self.temp_dir.name, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        data = {'text': ["I love data-juicer", "Today is a sunny day", "This is a test file"],
                'score': [9, 10, 11],
                'date': ["2025-05-20", "2025-05-21", "2025-05-22"]}
        df = pd.DataFrame(data)
        df.to_parquet(self.parquet_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parquet_to_jsonl(self):
        output_file = os.path.join(self.output_dir, 'test.jsonl')
        parquet_to_jsonl(self.parquet_file, output_file)

        self.assertTrue(os.path.exists(output_file))

        with open(output_file, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[0]),
                         {'text': "I love data-juicer", 'score': 9, 'date': "2025-05-20"})

    def test_get_parquet_file_names(self):
        files = get_parquet_file_names(self.temp_dir.name)
        self.assertIn(self.parquet_file, files)

    def test_convert_parquet_to_jsonl(self):
        convert_parquet_to_jsonl(self.parquet_file, self.output_dir)
        output_file = os.path.join(self.output_dir, 'test.jsonl')
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[1]),
                         {'text': "Today is a sunny day", 'score': 10, 'date': "2025-05-21"})


if __name__ == '__main__':
    unittest.main()
