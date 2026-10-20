import gzip
import tempfile
import unittest
from pathlib import Path

import zstandard as zstd

from data_juicer.utils.jsonl_lenient_loader import (
    dataset_from_lenient_jsonl_files,
    iter_lenient_jsonl_records,
)


class JsonlLenientLoaderEncodingTest(unittest.TestCase):
    def test_utf8_with_and_without_bom(self):
        for suffix in (".jsonl", ".jsonl.gz", ".jsonl.zst"):
            for encoding in ("utf-8", "utf-8-sig"):
                with self.subTest(suffix=suffix, encoding=encoding), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / ("records" + suffix)
                    payload = '{"text": "first"}\nnot json\n{"text": "second"}\n'.encode(encoding)
                    if suffix == ".jsonl.gz":
                        payload = gzip.compress(payload)
                    elif suffix == ".jsonl.zst":
                        payload = zstd.ZstdCompressor().compress(payload)
                    path.write_bytes(payload)

                    rows = list(iter_lenient_jsonl_records([(str(path), suffix)], add_suffix_column=False))

                    self.assertEqual(rows, [{"text": "first"}, {"text": "second"}])

    def test_dataset_preserves_first_record_with_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"text": "first"}\n{"text": "second"}\n', encoding="utf-8-sig")

            dataset = dataset_from_lenient_jsonl_files([(str(path), ".jsonl")], add_suffix_column=False)

            self.assertEqual(dataset.to_list(), [{"text": "first"}, {"text": "second"}])


if __name__ == "__main__":
    unittest.main()
