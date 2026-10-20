import gzip
import io
import os
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from data_juicer.download.arxiv import (
    ArxivDownloader,
    ArxivExtractor,
    ArxivIterator,
    download_arxiv,
)
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


class ArxivDownloaderTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_cached_file_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "src.tar")
            with open(path, "wb") as stream:
                stream.write(b"cached")
            with patch("data_juicer.download.arxiv.subprocess.run") as run:
                result = ArxivDownloader(tmpdir, verbose=True).download("src.tar")

        self.assertEqual(result, path)
        run.assert_not_called()

    @TEST_TAG("standalone")
    def test_download_command_quiet_verbose_and_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = ArxivDownloader(tmpdir)
            for verbose, returncode in ((False, 0), (True, 1)):
                with self.subTest(verbose=verbose):
                    downloader._verbose = verbose
                    with patch(
                        "data_juicer.download.arxiv.subprocess.run",
                        return_value=MagicMock(returncode=returncode),
                    ) as run:
                        output = io.StringIO()
                        with redirect_stdout(output):
                            result = downloader.download(f"src-{verbose}.tar")
                    kwargs = run.call_args.kwargs
                    expected = None if verbose else subprocess.DEVNULL
                    self.assertEqual(
                        run.call_args.args[0],
                        [
                            "s5cmd",
                            "--request-payer=requester",
                            "cp",
                            f"s3://arxiv/src/src-{verbose}.tar",
                            os.path.join(tmpdir, f"src-{verbose}.tar"),
                        ],
                    )
                    self.assertEqual(kwargs["stdout"], expected)
                    self.assertEqual(kwargs["stderr"], expected)
                    self.assertEqual(result, os.path.join(tmpdir, f"src-{verbose}.tar"))
                    if returncode:
                        self.assertIn("Failed to download", output.getvalue())


class ArxivIteratorTest(DataJuicerTestCaseBase):

    def _write_gzip(self, path, content):
        with gzip.open(path, "wb") as stream:
            stream.write(content)

    @TEST_TAG("standalone")
    def test_format_arxiv_id(self):
        iterator = ArxivIterator()
        self.assertEqual(iterator._format_arxiv_id("hep-th9901001"), "hep-th/9901001")
        self.assertEqual(iterator._format_arxiv_id("2401.12345"), "2401.12345")
        with self.assertRaises(ValueError):
            iterator._format_arxiv_id("bad_id")

    @TEST_TAG("standalone")
    def test_tex_project_loader_tar_gzip_and_invalid_inputs(self):
        iterator = ArxivIterator()
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "paper.tex")
            with open(tex_path, "w") as stream:
                stream.write("\\section{One} text")
            tar_path = os.path.join(tmpdir, "paper.tar")
            with tarfile.open(tar_path, "w") as archive:
                archive.add(tex_path, arcname="paper.tex")
            self.assertEqual(iterator._tex_proj_loader(tar_path), ["\\section{One} text"])

            gz_path = os.path.join(tmpdir, "paper.gz")
            self._write_gzip(gz_path, b"\\section{Two} text")
            self.assertEqual(iterator._tex_proj_loader(gz_path), ["\\section{Two} text"])

            binary_gz = os.path.join(tmpdir, "binary.gz")
            self._write_gzip(binary_gz, b"\xff")
            self.assertIsNone(iterator._tex_proj_loader(binary_gz))

            bad_path = os.path.join(tmpdir, "bad")
            with open(bad_path, "wb") as stream:
                stream.write(b"not compressed")
            self.assertIsNone(iterator._tex_proj_loader(bad_path))

    @TEST_TAG("standalone")
    def test_tex_project_loader_tar_decode_and_other_errors(self):
        iterator = ArxivIterator()
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = os.path.join(tmpdir, "binary.tex")
            with open(binary_path, "wb") as stream:
                stream.write(b"\xff")
            tar_path = os.path.join(tmpdir, "binary.tar")
            with tarfile.open(tar_path, "w") as archive:
                archive.add(binary_path, arcname="binary.tex")
            self.assertIsNone(iterator._tex_proj_loader(tar_path))

            with patch("data_juicer.download.arxiv.tarfile.open", side_effect=PermissionError("denied")):
                self.assertIsNone(iterator._tex_proj_loader(tar_path))

    @TEST_TAG("standalone")
    def test_iterate_archive_formats_ids_skips_bad_content_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, "projects")
            os.makedirs(project_dir)
            self._write_gzip(os.path.join(project_dir, "hep-th9901001.gz"), b"\\section{A} alpha")
            self._write_gzip(os.path.join(project_dir, "2401.12345.gz"), b"\\section{B} beta")
            self._write_gzip(os.path.join(project_dir, "bad_id.gz"), b"\\section{C} gamma")
            with open(os.path.join(project_dir, "broken"), "wb") as stream:
                stream.write(b"broken")

            outer_path = os.path.join(tmpdir, "snapshot.tar")
            with tarfile.open(outer_path, "w") as archive:
                for name in os.listdir(project_dir):
                    archive.add(os.path.join(project_dir, name), arcname=name)

            output = io.StringIO()
            with redirect_stdout(output):
                rows = list(ArxivIterator(log_frequency=1).iterate(outer_path))

        ids = {metadata["id"] for metadata, _ in rows}
        self.assertEqual(ids, {"hep-th/9901001", "2401.12345", "bad_id"})
        self.assertTrue(all(metadata["source_id"] == "snapshot.tar" for metadata, _ in rows))
        self.assertIn("failed to format arxiv id bad_id", output.getvalue())
        self.assertIn("Extracted", output.getvalue())


class ArxivExtractorTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_builds_both_non_argument_macro_styles(self):
        content = "\\newcommand*{\\foo}{Hello}\n\\def \\bar {World}\n"
        macros = ArxivExtractor()._build_non_arg_macros_dict(content)
        self.assertEqual(macros, {r"\\foo": "Hello", r"\\bar": "World"})

    @TEST_TAG("standalone")
    def test_clean_tex_removes_preamble_comments_and_references(self):
        extractor = ArxivExtractor()
        content = (
            "\\newcommand{\\foo}{World}\n"
            "preamble\n"
            "\\section[Short]{Intro}\n"
            "% whole line\n"
            "Hello \\foo! % inline\n"
            "Keep \\% literal\n"
            "\\appendix\nDrop"
        )
        macros = extractor._build_non_arg_macros_dict(content)
        cleaned = extractor._clean_tex_file(content, {r"\\takesarg": "unused"}, macros)
        self.assertTrue(cleaned.startswith("\\section[Short]{Intro}"))
        self.assertIn("Hello World", cleaned)
        self.assertIn(r"\% literal", cleaned)
        self.assertNotIn("preamble", cleaned)
        self.assertNotIn("whole line", cleaned)
        self.assertNotIn("Drop", cleaned)

    @TEST_TAG("standalone")
    def test_clean_tex_without_section_is_empty(self):
        self.assertEqual(ArxivExtractor()._clean_tex_file("plain text", {}, {}), "")

    @TEST_TAG("standalone")
    def test_extract_empty_valid_and_cleaning_error(self):
        extractor = ArxivExtractor()
        self.assertIsNone(extractor.extract([]))
        metadata, text = extractor.extract(
            ["\\newcommand{\\name}{Arxiv}\n\\section{Intro} Hello \\name!", "no section"]
        )
        self.assertEqual(metadata, {})
        self.assertIn("Hello Arxiv", text)

        with patch.object(extractor, "_clean_tex_file", side_effect=RuntimeError("bad tex")):
            self.assertEqual(extractor.extract(["content"]), ({}, None))

    @TEST_TAG("standalone")
    def test_extract_returns_none_when_all_files_clean_to_empty(self):
        self.assertIsNone(ArxivExtractor().extract(["preamble only"]))


class DownloadArxivTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_download_arxiv_builds_controlled_pipeline(self):
        sentinel = object()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("data_juicer.download.arxiv.get_arxiv_urls", return_value=["a.tar", "b.tar", "c.tar"]),
            patch("data_juicer.download.arxiv.expand_outdir_and_mkdir") as mkdir,
            patch("data_juicer.download.arxiv.download_and_extract", return_value=sentinel) as run,
        ):
            result = download_arxiv(
                tmpdir,
                output_type="parquet",
                url_limit=2,
                keep_raw_download=True,
                force_download=True,
            )

        self.assertIs(result, sentinel)
        mkdir.assert_called_once_with(os.path.join(tmpdir, "downloads"))
        args = run.call_args.args
        self.assertEqual(args[0], ["a.tar", "b.tar"])
        self.assertEqual(
            args[1],
            [os.path.join(tmpdir, "a.tar.parquet"), os.path.join(tmpdir, "b.tar.parquet")],
        )
        self.assertIsInstance(args[2], ArxivDownloader)
        self.assertIsInstance(args[3], ArxivIterator)
        self.assertIsInstance(args[4], ArxivExtractor)
        self.assertEqual(args[5]["text"], str)
        self.assertEqual(run.call_args.kwargs["output_type"], "parquet")
        self.assertTrue(run.call_args.kwargs["keep_raw_download"])
        self.assertTrue(run.call_args.kwargs["force_download"])

    @TEST_TAG("standalone")
    def test_download_arxiv_respects_explicit_raw_directory_without_limit(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("data_juicer.download.arxiv.get_arxiv_urls", return_value=["one.tar"]),
            patch("data_juicer.download.arxiv.expand_outdir_and_mkdir") as mkdir,
            patch("data_juicer.download.arxiv.download_and_extract", return_value=MagicMock()) as run,
        ):
            raw_dir = os.path.join(tmpdir, "raw")
            download_arxiv(tmpdir, raw_download_dir=raw_dir, url_limit=0)

        mkdir.assert_called_once_with(raw_dir)
        self.assertEqual(run.call_args.args[0], ["one.tar"])
        self.assertEqual(run.call_args.args[2]._download_dir, raw_dir)


if __name__ == "__main__":
    unittest.main()
