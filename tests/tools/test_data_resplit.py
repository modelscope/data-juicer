import os
import tempfile
import unittest
import json
from tools.data_resplit import split_jsonl, get_jsonl_file_names


class TestDataResplit(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.jsonl_file = os.path.join(self.temp_dir.name, 'test.jsonl')
        self.output_dir = os.path.join(self.temp_dir.name, 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        data = [{"id": i, "text": f"Data Juicer Sample text {i}"} for i in range(10)]
        # Oversize line
        data.append({"id": 10, "text": "This is a Oversize line" * 1000})
        with open(self.jsonl_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_split_jsonl(self):
        max_size = 0.001
        split_jsonl(self.jsonl_file, max_size, self.output_dir)

        output_files = [f for f in os.listdir(self.output_dir) if f.endswith('.jsonl')]
        self.assertGreater(len(output_files), 1)

        total_records = 0
        for file in output_files:
            with open(os.path.join(self.output_dir, file), 'r') as f:
                lines = f.readlines()
                total_records += len(lines)
        self.assertEqual(total_records, 11)

    def test_get_jsonl_file_names(self):
        files = get_jsonl_file_names(self.temp_dir.name)
        self.assertIn(self.jsonl_file, files)


if __name__ == '__main__':
    unittest.main()
