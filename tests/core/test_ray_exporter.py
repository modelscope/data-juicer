import copy
import os
import os.path as osp
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.utils.constant import Fields, HashKeys
from data_juicer.utils.mm_utils import load_images_byte
from tests.core.test_exporter import IcebergWriterFixture


class TestRayExporter(DataJuicerTestCaseBase):

    def setUp(self):
        """Set up test data"""
        super().setUp()

        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        cur_dir = osp.dirname(osp.abspath(__file__))
        self.tmp_dir = f'{cur_dir}/tmp/{self.__class__.__name__}/{self._testMethodName}'
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.data = [
            {'text': 'hello', Fields.stats: {'score': 1}, HashKeys.hash: 'a1'},
            {'text': 'world', Fields.stats: {'score': 2}, HashKeys.hash: 'b2'},
            {'text': 'test', Fields.stats: {'score': 3}, HashKeys.hash: 'c3'}
        ]
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up temporary outputs"""

        self.dataset = None
        if osp.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

        super().tearDown()

    def _pop_raw_data_keys(self, keys):
        res = copy.deepcopy(self.data)
        for d_i in res:
            for k in keys:
                d_i.pop(k, None)

        return res

    @TEST_TAG('ray')
    def test_json_not_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.json')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats, HashKeys.hash]))

    @TEST_TAG('ray')
    def test_jsonl_keep_stats_and_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.jsonl')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_json(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self.data)

    @TEST_TAG('ray')
    def test_parquet_keep_stats(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.parquet')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=True,
            keep_hashes_in_res_ds=False)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_parquet(out_path)
        data_list = ds.take_all()

        self.assertListEqual(data_list, self._pop_raw_data_keys([HashKeys.hash]))

    @TEST_TAG('ray')
    def test_lance_keep_hashes(self):
        import ray

        out_path = osp.join(self.tmp_dir, 'outdata.lance')
        ray_exporter = RayExporter(
            out_path,
            keep_stats_in_res_ds=False,
            keep_hashes_in_res_ds=True)
        ray_exporter.export(self.dataset.data)

        ds = ray.data.read_lance(out_path)
        data_list = ds.take_all()

        self.assertListOfDictEqual(data_list, self._pop_raw_data_keys([Fields.stats]))

    @TEST_TAG('ray')
    def test_webdataset_multi_images(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'images': [img1_path, img2_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path])},
            {
                'json': {
                    'text': 'world',
                    'images': [img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img2_path, img3_path])},
            {
                'json': {
                    'text': 'test',
                    'images': [img1_path, img2_path, img3_path]
                    },
                'jpgs': load_images_byte([img1_path, img2_path, img3_path])}
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path)
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])

        for i in range(len(data)):
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(
                res_list[i]['jpgs'],
                [Image.open(io.BytesIO(v)) for v in data[i]['jpgs']]
            )

    @TEST_TAG('ray')
    def test_webdataset_multi_videos_frames_bytes(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img1.png')
        img2_path = osp.join(data_dir, 'img2.jpg')
        img3_path = osp.join(data_dir, 'img3.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'videos': ['video1.mp4', 'video2.mp4']
                    },
                'mp4s': [
                    load_images_byte([img1_path]),  # as video1 frames bytes
                    load_images_byte([img1_path, img2_path])   # as video2 frames path
                    ]
            },
            {
                'json': {
                    'text': 'world',
                    'videos': ['video1.mp4']
                    },
                'mp4s': [
                    load_images_byte([img2_path, img3_path])  # as video1 frames
                    ]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp4s']) > 1:
                tgt_mp4s = [[Image.open(io.BytesIO(f_i)) for f_i in v_i] for v_i in data[i]['mp4s']]
            else:
                tgt_mp4s = [Image.open(io.BytesIO(f_i)) for f_i in data[i]['mp4s'][0]]
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(res_list[i]['mp4s'], tgt_mp4s)

    @TEST_TAG('ray')
    def test_webdataset_multi_videos_frames_path(self):
        import io
        from PIL import Image
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        img1_path = osp.join(data_dir, 'img8.jpg')
        img2_path = osp.join(data_dir, 'img9.jpg')
        img3_path = osp.join(data_dir, 'img10.jpg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    'videos': ['video1.mp4', 'video2.mp4']
                    },
                'mp4s': [
                    [img1_path],  # as video1 frames path
                    [img1_path, img2_path]   # as video2 frames path
                    ]
            },
            {
                'json': {
                    'text': 'world',
                    'videos': ['video1.mp4']
                    },
                'mp4s': [
                    [img2_path, img3_path]  # as video1 frames path
                    ]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))
        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp4s']) > 1:
                tgt_mp4s = [[Image.open(f_i, formats=['jpeg']) for f_i in v_i] for v_i in data[i]['mp4s']]
            else:
                tgt_mp4s = [Image.open(f_i, formats=['jpeg']) for f_i in data[i]['mp4s'][0]]
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])
            self.assertEqual(res_list[i]['mp4s'], tgt_mp4s)

    @TEST_TAG('ray')
    def test_webdataset_multi_audios_path(self):
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.utils.mm_utils import load_audio

        data_dir = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '..', 'ops', 'data'))
        audio1_path = osp.join(data_dir, 'audio1.wav')
        audio2_path = osp.join(data_dir, 'audio2.wav')
        audio3_path = osp.join(data_dir, 'audio3.ogg')

        data = [
            {
                'json': {
                    'text': 'hello',
                    },
                'mp3s': [audio1_path]
            },
            {
                'json': {
                    'text': 'world',
                    },
                'mp3s': [audio2_path, audio3_path]
            }
        ]
        dataset = RayDataset(ray.data.from_items(data))
        out_path = osp.join(self.tmp_dir, 'outdata.webdataset')
        ray_exporter = RayExporter(out_path, export_type='webdataset')
        ray_exporter.export(dataset.data)

        ds = RayDataset.read_webdataset(out_path)
        res_list = ds.take_all()
        
        self.assertEqual(len(res_list), len(data))

        res_list.sort(key=lambda x: x['json']['text'])
        data.sort(key=lambda x: x['json']['text'])
        
        for i in range(len(data)):
            if len(data[i]['mp3s']) <= 1:
                mp3s_list = [res_list[i]['mp3s']]
            else:
                mp3s_list = res_list[i]['mp3s']

            tgt_mp3s = [load_audio(f_i) for f_i in data[i]['mp3s']]
            
            self.assertDictEqual(res_list[i]['json'], data[i]['json'])

            for j in range(len(mp3s_list)):
                arr, sampling_rate = mp3s_list[j]
                tgt_arr, tgt_sampling_rate = tgt_mp3s[j]
                import numpy as np
                np.testing.assert_array_equal(arr, tgt_arr)
                self.assertEqual(sampling_rate, tgt_sampling_rate)


class RayExporterEncryptTest(DataJuicerTestCaseBase):
    """Unit tests for RayExporter encryption features (no Ray dependency)."""

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()
        # Generate a fresh Fernet key for every test
        from cryptography.fernet import Fernet
        self._key = Fernet.generate_key()
        self._key_file = os.path.join(self.tmp_dir, 'test.key')
        with open(self._key_file, 'wb') as f:
            f.write(self._key)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # __init__ – encryption flag and Fernet setup
    # ------------------------------------------------------------------

    def test_encrypt_disabled_by_default(self):
        """encrypt_before_export defaults to False; _fernet stays None."""
        exporter = RayExporter(export_path=os.path.join(self.tmp_dir, 'out.jsonl'))
        self.assertFalse(exporter.encrypt_before_export)
        self.assertIsNone(exporter._fernet)

    def test_encrypt_enabled_loads_fernet_key(self):
        """encrypt_before_export=True with a valid key file sets _fernet."""
        exporter = RayExporter(
            export_path=os.path.join(self.tmp_dir, 'out.jsonl'),
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )
        self.assertTrue(exporter.encrypt_before_export)
        self.assertIsNotNone(exporter._fernet)

    @patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem")
    def test_s3_path_disables_encryption_with_warning(self, mock_create_s3_fs):
        """S3 export path silently disables local-file encryption."""
        warnings_seen = []
        from loguru import logger as _logger
        handler_id = _logger.add(
            lambda msg: warnings_seen.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = RayExporter(
                export_path='s3://some-bucket/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=self._key_file,
            )
        finally:
            _logger.remove(handler_id)
        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            any('encrypt_before_export' in str(w) and 's3' in str(w).lower()
                for w in warnings_seen),
            'Expected warning about S3 + encrypt_before_export not found',
        )

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_hdfs_path_disables_encryption_with_warning(self, mock_create_hdfs_fs):
        """HDFS export path silently disables local-file encryption."""
        warnings_seen = []
        from loguru import logger as _logger
        handler_id = _logger.add(
            lambda msg: warnings_seen.append(str(msg)),
            level='WARNING',
            format='{message}',
        )
        try:
            exporter = RayExporter(
                export_path='hdfs://namenode:8020/user/data/out.jsonl',
                encrypt_before_export=True,
                encryption_key_path=self._key_file,
            )
        finally:
            _logger.remove(handler_id)
        self.assertFalse(exporter.encrypt_before_export)
        self.assertTrue(
            any('encrypt_before_export' in str(w) for w in warnings_seen),
            'Expected warning about HDFS + encrypt_before_export not found',
        )

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    @patch("data_juicer.core.ray_exporter.is_remote_path")
    def test_hdfs_path_skips_local_makedirs(self, mock_is_remote, mock_create_hdfs_fs):
        """HDFS export_path should skip local os.makedirs via is_remote_path."""
        mock_is_remote.return_value = True

        exporter = RayExporter(
            export_path='hdfs://namenode:8020/user/data',
            export_type='jsonl',
        )

        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ['text']
        mock_dataset.count.return_value = 1
        mock_dataset.size_bytes.return_value = 50
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, '_router') as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {'jsonl': mock_write}
            with patch("data_juicer.core.ray_exporter.os.makedirs") as mock_makedirs:
                exporter._export_impl(mock_dataset, 'hdfs://namenode:8020/user/data')

        # os.makedirs should NOT be called because is_remote_path returned True
        mock_makedirs.assert_not_called()


    # ------------------------------------------------------------------
    # _export_impl – encrypt_file called for each file in the output dir
    # ------------------------------------------------------------------

    def test_export_impl_encrypts_output_files(self):
        """After _export_impl, all files inside the output dir are encrypted."""
        from cryptography.fernet import Fernet
        from data_juicer.utils.encryption_utils import decrypt_file_to_bytes

        export_dir = os.path.join(self.tmp_dir, 'export_out')
        exporter = RayExporter(
            export_path=export_dir,
            export_type='jsonl',
            encrypt_before_export=True,
            encryption_key_path=self._key_file,
        )

        # Write two plain-text sentinel files to simulate Ray output
        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "hello"}\n'
        for fname in ('part-0.jsonl', 'part-1.jsonl'):
            fpath = os.path.join(export_dir, fname)
            with open(fpath, 'wb') as f:
                f.write(sentinel)

        # Mock the underlying Ray write call so _export_impl only runs
        # the post-write encryption step
        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ['text']
        mock_dataset.count.return_value = 2
        mock_dataset.size_bytes.return_value = 100
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, '_router') as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {'jsonl': mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        # Verify every file is now encrypted (not plain text any more)
        fernet = Fernet(self._key)
        for fname in ('part-0.jsonl', 'part-1.jsonl'):
            fpath = os.path.join(export_dir, fname)
            raw = open(fpath, 'rb').read()
            self.assertNotEqual(raw, sentinel,
                                f'{fname} should have been encrypted')
            # Must be decryptable with the same key
            decrypted = decrypt_file_to_bytes(fpath, fernet)
            self.assertEqual(decrypted, sentinel)

    def test_export_impl_skips_encryption_when_disabled(self):
        """When encrypt_before_export=False, files are left as-is."""
        export_dir = os.path.join(self.tmp_dir, 'plain_out')
        exporter = RayExporter(
            export_path=export_dir,
            export_type='jsonl',
            encrypt_before_export=False,
        )

        os.makedirs(export_dir, exist_ok=True)
        sentinel = b'{"text": "world"}\n'
        fpath = os.path.join(export_dir, 'part-0.jsonl')
        with open(fpath, 'wb') as f:
            f.write(sentinel)

        mock_dataset = MagicMock()
        mock_dataset.columns.return_value = ['text']
        mock_dataset.count.return_value = 1
        mock_dataset.size_bytes.return_value = 50
        mock_dataset.drop_columns.return_value = mock_dataset

        with patch.object(RayExporter, '_router') as mock_router:
            mock_write = MagicMock(return_value=None)
            mock_router.return_value = {'jsonl': mock_write}
            exporter._export_impl(mock_dataset, export_dir)

        self.assertEqual(open(fpath, 'rb').read(), sentinel,
                         'File should not be modified when encryption is off')


class RayIcebergExporterConfigTest(IcebergWriterFixture, unittest.TestCase):
    def test_invalid_configuration_and_old_ray(self):
        invalid = [
            {"mode": "upsert"}, {"mode": None}, {"table_identifier": ""}, {"catalog_kwargs": None},
            {"catalog_kwargs": {}}, {"catalog_kwargs": {"name": " "}},
            {"catalog_kwargs": {k: v for k, v in self.catalog_kwargs.items() if k != "name"}},
            {"export_path": "file:///tmp/output"},
            {"export_path": None}, {"export_path": ""}, {"export_path": "s3://bucket/output"},
            {"export_path": "hdfs://host/output"}, {"export_path": "https://host/output"},
            {"export_shard_size": 1},
            {"encrypt_before_export": True, "encryption_key_path": "/must/not/read"},
            {"overwrite_filter": "id > 0"}, {"overwrite_kwargs": {"branch": "other"}},
        ]
        with patch("data_juicer.utils.encryption_utils.load_fernet_key") as key_loader, \
                patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem") as filesystem:
            for changes in invalid:
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    RayExporter(**dict(self.options(), **changes))
            key_loader.assert_not_called()
            filesystem.assert_not_called()
        with patch("ray.__version__", "2.51.0"), self.assertRaisesRegex(RuntimeError, "Ray >=2.52.0"):
            RayExporter(**self.options())
        writer = RayExporter(**self.options())
        self.assertEqual(writer.iceberg_config["mode"], "overwrite")
        self.catalog_kwargs["warehouse"] = "changed-after-construction"
        self.assertNotEqual(writer.iceberg_config["catalog_kwargs"]["warehouse"], self.catalog_kwargs["warehouse"])

    def test_materialize_once_before_column_probe(self):
        source = MagicMock()
        cached = source.materialize.return_value

        def columns():
            source.materialize.assert_called_once_with()
            return ["id", "text", Fields.stats]

        cached.columns.side_effect = columns
        writer = RayExporter(**self.options(keep_stats_in_res_ds=False))
        with patch.object(writer, "write_iceberg") as write:
            writer.export(source, columns=["stale"])
        cached.columns.assert_called_once_with()
        cached.drop_columns.assert_called_once_with([Fields.stats])
        write.assert_called_once_with(cached.drop_columns.return_value, self.anchor, **writer.iceberg_config)
        source.columns.assert_not_called()
        cached.materialize.assert_not_called()

    def test_windows_artifact_constructor(self):
        writer = RayExporter(**dict(self.options(), export_path="C:/tmp/result.jsonl"))
        self.assertEqual(writer.export_path, "C:/tmp/result.jsonl")

    def test_executor_iceberg_uses_configured_stats_flags(self):
        from types import SimpleNamespace
        from data_juicer.core.executor.ray_executor import RayExecutor

        cfg = SimpleNamespace(
            export_path=self.anchor, export_type="iceberg", export_shard_size=0,
            export_extra_args={"table_identifier": self.identifier, "catalog_kwargs": self.catalog_kwargs},
            keep_stats_in_res_ds=False, keep_hashes_in_res_ds=False, work_dir=str(self.root),
            open_tracer=False, min_common_dep_num_to_combine=-1,
        )
        module = "data_juicer.core.executor.ray_executor"
        with patch(f"{module}.ExecutorBase.__init__", lambda obj, config: setattr(obj, "cfg", config)), \
                patch(f"{module}.EventLoggingMixin.__init__", return_value=None), \
                patch(f"{module}.DAGExecutionMixin.__init__", return_value=None), \
                patch(f"{module}.DatasetBuilder"), patch(f"{module}.ray"), \
                patch("data_juicer.utils.ray_utils.initialize_ray"):
            executor = RayExecutor(cfg)
        self.assertFalse(executor.exporter.keep_stats_in_res_ds)
        source = MagicMock()
        source.materialize.return_value.columns.return_value = ["id", "text", Fields.stats, Fields.meta]
        with patch.object(executor.exporter, "write_iceberg"):
            executor.exporter.export(source)
        self.assertCountEqual(source.materialize.return_value.drop_columns.call_args.args[0],
                              [Fields.stats, Fields.meta])

    def test_native_file_collision_and_isolated_attempts(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink
        from ray.data._internal.execution.interfaces import TaskContext
        from data_juicer.utils.iceberg_utils import iceberg_datasink

        sink = iceberg_datasink(self.identifier, self.catalog_kwargs, "append")
        sink.on_write_start()
        shared_uuid = sink._write_uuid
        blocks = [pa.Table.from_pylist([row], schema=self.schema) for row in self.rows]
        ctx = TaskContext(task_idx=0, op_name="iceberg-test")
        location = self.table.metadata_location
        # Reproduce Ray 2.52's native counter reset without schema commits.
        broken = IcebergDatasink.write(sink, iter(blocks), ctx)
        self.assertEqual(len(broken), 2)
        self.assertEqual(len({f.file_path for f in broken}), 1)
        self.assertEqual(pq.read_table(broken[0].file_path).to_pylist(), self.rows[-1:])
        first = sink.write(iter(blocks), ctx)
        retry = sink.write(iter(blocks), ctx)
        other_task = sink.write(iter(blocks), TaskContext(task_idx=1, op_name="iceberg-test"))
        files = first + retry + other_task
        self.assertEqual(len({f.file_path for f in files}), 6)
        self.assertCountEqual([row for f in files for row in pq.read_table(f.file_path).to_pylist()],
                              self.rows * 3)
        self.assertEqual(sink._write_uuid, shared_uuid)
        bad = pa.table({"id": [3], "text": ["bad"], "extra": [1]})
        with self.assertRaisesRegex(ValueError, "Schema mismatch"):
            sink.write(iter([blocks[0], bad]), ctx)
        self.assertEqual(sink._write_uuid, shared_uuid)
        self.assertEqual(self.table.refresh().metadata_location, location)
        self.assertEqual(self.read_rows(), self.initial)

    def test_sink_worker_validation_never_commits_schema(self):
        import pyarrow as pa
        from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink
        from ray.data._internal.savemode import SaveMode
        from data_juicer.utils.iceberg_utils import iceberg_datasink

        native_update = IcebergDatasink._update_schema
        before = self.snapshot()
        location = self.table.metadata_location
        for mode, expected in (("append", SaveMode.APPEND), ("overwrite", SaveMode.OVERWRITE)):
            sink = iceberg_datasink(self.identifier, self.catalog_kwargs, mode)
            self.assertIsInstance(sink, IcebergDatasink)
            self.assertEqual(sink._mode, expected)
            sink._update_schema(self.schema)
            documented = pa.schema([pa.field("id", pa.int64(), metadata={b"doc": b"new doc"}),
                                    ("text", pa.string())])
            with self.assertRaisesRegex(ValueError, "automatic schema evolution"):
                sink._update_schema(documented)
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.table.metadata_location, location)
        self.assertIs(IcebergDatasink._update_schema, native_update)
        self.assertEqual(self.catalog_kwargs["name"], "local_writer_test")

    def test_sink_rejects_incompatible_ray_internals(self):
        from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink
        from data_juicer.utils.iceberg_utils import iceberg_datasink

        with patch.object(IcebergDatasink, "_reload_table", None):
            with self.assertRaisesRegex(RuntimeError, "IcebergDatasink API is unsupported"):
                iceberg_datasink(self.identifier, self.catalog_kwargs, "append")


class RayIcebergExporterTest(IcebergWriterFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray
        from packaging.version import Version

        if Version(ray.__version__) < Version("2.52.0"):
            raise unittest.SkipTest("Iceberg writing requires Ray >=2.52.0")
        cls.owns_ray = not ray.is_initialized()
        if cls.owns_ray:
            # Isolated local instance; never stop another user's Ray cluster.
            cls.ray_temp = tempfile.TemporaryDirectory(prefix="dj_ray_iceberg_")
            ray.init(address="local", num_cpus=2, include_dashboard=False,
                     object_store_memory=128 * 1024 * 1024, _temp_dir=cls.ray_temp.name)
        from ray.data import DataContext

        cls.old_progress = DataContext.get_current().enable_progress_bars
        DataContext.get_current().enable_progress_bars = False

    @classmethod
    def tearDownClass(cls):
        import ray
        from ray.data import DataContext

        DataContext.get_current().enable_progress_bars = cls.old_progress
        if cls.owns_ray:
            ray.shutdown()
            cls.ray_temp.cleanup()

    @TEST_TAG("ray")
    def test_native_append_overwrite_repeated_and_empty(self):
        import pyarrow as pa
        import ray

        arrow = pa.Table.from_pylist(self.rows, schema=self.schema)
        for mode in ("append", "overwrite"):
            with self.subTest(mode=mode):
                self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
                writer = RayExporter(**self.options(mode=mode))
                ds = ray.data.from_arrow(arrow)
                writer.export(ds)
                writer.export(ds)
                expected = self.initial + self.rows * 2 if mode == "append" else self.rows
                self.assertCountEqual(self.read_rows(), expected)
                for empty in (ray.data.from_arrow(arrow.slice(0, 0)), ray.data.from_items([])):
                    if mode == "overwrite":
                        self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
                    before = self.snapshot()
                    writer.export(empty)
                    if mode == "append":
                        self.assertCountEqual(self.read_rows(), expected)
                        self.assertEqual(self.snapshot(), before)
                    else:
                        self.assertEqual(self.read_rows(), [])
                        self.assertNotEqual(self.snapshot(), before)
                        self.assertEqual(self.table.current_snapshot().parent_snapshot_id, before)
        self.assertFalse(os.path.exists(self.anchor))
        self.assertFalse(os.path.exists(os.path.dirname(self.anchor)))

    @TEST_TAG("ray")
    def test_required_columns_accept_nullable_blocks_without_nulls(self):
        import pyarrow as pa
        import ray

        schema = pa.schema([pa.field("id", pa.int64(), nullable=False), ("text", pa.string())])
        table = self.catalog.create_table("training.required", schema=schema)
        required_schema = table.schema()
        writer = RayExporter(**dict(self.options(), table_identifier="training.required"))
        arrow = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.assertTrue(arrow.schema.field("id").nullable)
        writer.export(ray.data.from_arrow(arrow))
        self.assertCountEqual(table.refresh().scan().to_arrow().to_pylist(), self.rows)
        # Native union-by-name would have relaxed the required id.
        self.assertEqual(table.schema(), required_schema)
        self.assertEqual(table.schema().schema_id, required_schema.schema_id)
        snapshot = table.current_snapshot().snapshot_id
        nulls = pa.Table.from_pylist([{"id": None, "text": "missing"}], schema=self.schema)
        with self.assertRaisesRegex(Exception, "required column id contains null"):
            writer.export(ray.data.from_arrow(nulls))
        self.assertEqual(table.refresh().current_snapshot().snapshot_id, snapshot)

    @TEST_TAG("ray")
    def test_multiblock_two_workers_multiset_and_file_paths(self):
        from collections import Counter
        import pyarrow as pa
        import pyarrow.parquet as pq
        import ray

        @ray.remote(num_cpus=0)
        class ExecutionCounter:
            def __init__(self):
                self.calls = 0

            def increment(self):
                self.calls += 1

            def count(self):
                return self.calls

        counter = ExecutionCounter.remote()
        self.addCleanup(ray.kill, counter)
        native_write = ray.data.Dataset.write_datasink
        committed = []

        def transform(batch):
            ray.get(counter.increment.remote())
            return batch

        def write(dataset, sink):
            original_complete = sink.on_write_complete
            native_state = type(sink).__getstate__

            def state(obj):
                result = native_state(obj)
                result.pop("on_write_complete", None)  # Driver-only assertions.
                return result

            def complete(result):
                self.assertEqual(result.num_rows, 1024)
                self.assertEqual(len(result.write_returns), 2)
                self.assertEqual([len(files) for files in result.write_returns], [4, 4])
                files = [f for group in result.write_returns for f in group]
                paths = {f.file_path for f in files}
                self.assertEqual(len(paths), 8)
                physical = [row for f in files for row in pq.read_table(f.file_path).to_pylist()]
                self.assertEqual(Counter((r["id"], r["text"]) for r in physical),
                                 Counter((r["id"], r["text"]) for r in rows))
                original_complete(result)
                committed.append(paths)

            sink.on_write_complete = complete
            # Only this locally-created subclass: force four blocks per native task.
            with patch.object(type(sink), "min_rows_per_write", new=property(lambda obj: 512)), \
                    patch.object(type(sink), "__getstate__", new=state):
                return native_write(dataset, sink, concurrency=2)

        for mode in ("overwrite", "append"):
            # Overwrite covers unique IDs; append covers legitimate repeated rows/IDs.
            ids = [i if mode == "overwrite" else i // 2 for i in range(1024)]
            rows = [{"id": i, "text": f"row-{i}"} for i in ids]
            arrow = pa.Table.from_pylist(rows, schema=self.schema)
            self.table.overwrite(pa.Table.from_pylist(self.initial, schema=self.schema))
            initial_paths = {task.file.file_path for task in self.table.scan().plan_files()}
            writer = RayExporter(**self.options(mode=mode))
            for run in range(2):
                ds = ray.data.from_arrow([arrow.slice(i * 128, 128) for i in range(8)])
                ds = ds.map_batches(transform, batch_format="pyarrow", batch_size=None)
                before = ray.get(counter.count.remote())
                with patch.object(ray.data.Dataset, "write_datasink", autospec=True, side_effect=write):
                    writer.export(ds)
                self.assertEqual(ray.get(counter.count.remote()) - before, 8)
                expected = self.initial + rows * (run + 1) if mode == "append" else rows
                actual = self.read_rows()
                self.assertEqual(Counter((r["id"], r["text"]) for r in actual),
                                 Counter((r["id"], r["text"]) for r in expected))
                self.assertEqual(len({r["id"] for r in actual}), 1024 if mode == "overwrite" else 512)
                self.assertEqual(len(actual), 1024 if mode == "overwrite" else 1 + 1024 * (run + 1))
                tasks = list(self.table.scan().plan_files())
                active_paths = [task.file.file_path for task in tasks]
                self.assertEqual(len(active_paths), len(set(active_paths)))
                expected_paths = set.union(*committed[-(run + 1):]) if mode == "append" else committed[-1]
                if mode == "append":
                    expected_paths |= initial_paths
                self.assertEqual(set(active_paths), expected_paths)
        self.assertEqual(len(set.union(*committed)), 32)
        self.assertFalse(os.path.exists(self.anchor))

    @TEST_TAG("ray")
    def test_cached_execution_and_custom_native_datasink(self):
        import pyarrow as pa
        import ray
        from ray.data._internal.datasource.iceberg_datasink import IcebergDatasink
        from ray.data._internal.savemode import SaveMode

        @ray.remote(num_cpus=0)
        class Counter:
            def __init__(self):
                self.calls = 0

            def increment(self):
                self.calls += 1
                return self.calls

            def count(self):
                return self.calls

        counter = Counter.remote()
        self.addCleanup(ray.kill, counter)

        def transform(batch):
            value = ray.get(counter.increment.remote())
            return batch.set_column(0, "id", pa.array([value] * len(batch), type=pa.int64()))

        arrow = pa.Table.from_pylist(self.rows, schema=self.schema)
        ds = ray.data.from_arrow(arrow).map_batches(transform, batch_format="pyarrow", batch_size=None)
        native_write = ray.data.Dataset.write_datasink

        def write(dataset, sink):
            self.assertIsInstance(sink, IcebergDatasink)
            self.assertIsNot(type(sink), IcebergDatasink)
            self.assertEqual(sink._mode, SaveMode.OVERWRITE)
            return native_write(dataset, sink)

        with patch.object(ray.data.Dataset, "write_datasink", autospec=True, side_effect=write) as routed:
            RayExporter(**self.options()).export(ds)
        routed.assert_called_once()
        self.assertEqual(ray.get(counter.count.remote()), 1)
        self.assertCountEqual(self.read_rows(), [{"id": 1, "text": row["text"]} for row in self.rows])

    @TEST_TAG("ray")
    def test_stats_hashes_use_actual_columns(self):
        import pyarrow as pa
        import ray

        rows = [dict(row, **{Fields.stats: {"score": i}, Fields.meta: {"source": "local"},
                            HashKeys.hash: f"h{i}"}) for i, row in enumerate(self.rows)]
        arrow = pa.Table.from_pylist(rows)
        for keep_stats in (False, True):
            for keep_hashes in (False, True):
                with self.subTest(stats=keep_stats, hashes=keep_hashes):
                    columns = ["id", "text"]
                    if keep_stats:
                        columns += [Fields.stats, Fields.meta]
                    if keep_hashes:
                        columns += [HashKeys.hash]
                    identifier = f"training.kept_{int(keep_stats)}_{int(keep_hashes)}"
                    table = self.catalog.create_table(identifier, schema=arrow.select(columns).schema)
                    writer = RayExporter(**dict(self.options(), table_identifier=identifier),
                                         keep_stats_in_res_ds=keep_stats, keep_hashes_in_res_ds=keep_hashes)
                    # Executor input columns omit columns produced by operators.
                    writer.export(ray.data.from_arrow(arrow), columns=["id", "text"])
                    self.assertCountEqual(table.refresh().scan().to_arrow().to_pylist(),
                                          arrow.select(columns).to_pylist())
        self.assertFalse(os.path.exists(self.anchor))

    @TEST_TAG("ray")
    def test_rich_types_native_roundtrip(self):
        import ray

        arrow = self.rich_arrow()
        writer = RayExporter(**dict(self.options(), table_identifier="training.typed"))
        writer.export(ray.data.from_arrow(arrow))
        result = self.catalog.load_table("training.typed").scan().to_arrow()
        self.assertCountEqual(result.to_pylist(), arrow.to_pylist())

    @TEST_TAG("ray")
    def test_native_read_then_write_existing_table(self):
        import ray

        writer = RayExporter(**self.options())
        for _ in range(2):
            ds = ray.data.read_iceberg(table_identifier=self.identifier, catalog_kwargs=copy.deepcopy(self.catalog_kwargs))
            writer.export(ds)
            self.assertEqual(self.read_rows(), self.initial)

    @TEST_TAG("ray")
    def test_schema_error_precedes_sink_without_snapshot_change(self):
        import pyarrow as pa
        import ray

        before = self.snapshot()
        location = self.table.metadata_location
        schema_id = self.table.schema().schema_id
        writer = RayExporter(**self.options())
        for arrow in (pa.table({"id": ["wrong"], "text": ["x"]}),
                      pa.table({"id": [1], "text": ["x"], "extra": [2]})):
            with self.subTest(schema=arrow.schema), self.assertRaises(ValueError):
                writer.export(ray.data.from_arrow(arrow))
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(self.table.metadata_location, location)
            self.assertEqual(self.table.schema().schema_id, schema_id)
        # A good first block must not hide an incompatible later block.
        mixed = ray.data.from_arrow([
            pa.Table.from_pylist(self.rows, schema=self.schema),
            pa.table({"id": [3], "text": ["bad"], "extra": [2]}),
        ])
        native_write = ray.data.Dataset.write_datasink

        def serial_write(dataset, sink):
            # A failing block cancels sibling tasks. Interrupting a sibling's
            # first SQLAlchemy import leaves cached submodules without their
            # parent attributes and poisons that worker for subsequent tests.
            # Still validate the later block, but without an importing sibling.
            return native_write(dataset, sink, concurrency=1)

        with patch.object(ray.data.Dataset, "write_datasink", autospec=True, side_effect=serial_write), \
                self.assertRaisesRegex(ValueError, "more columns: extra"):
            writer.export(mixed)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.table.metadata_location, location)
        self.assertEqual(self.read_rows(), self.initial)
        self.assertFalse(os.path.exists(self.anchor))

    @TEST_TAG("ray")
    def test_schema_evolution_is_rejected_without_metadata_commit(self):
        import pyarrow as pa
        import ray

        # A column doc can cause evolution even when types are compatible.
        documented = pa.schema([pa.field("id", pa.int64(), metadata={b"doc": b"new doc"}),
                                ("text", pa.string())])
        before = self.snapshot()
        location = self.table.metadata_location
        writer = RayExporter(**self.options())
        ds = ray.data.from_arrow(pa.Table.from_pylist(self.rows, schema=documented))
        with self.assertRaisesRegex(ValueError, "automatic schema evolution"):
            writer.export(ds)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.table.metadata_location, location)

    @TEST_TAG("ray")
    def test_missing_table_even_empty_no_fallback(self):
        import pyarrow as pa
        import ray

        for mode in ("append", "overwrite"):
            writer = RayExporter(**dict(self.options(mode=mode), table_identifier="training.missing"))
            for ds in (ray.data.from_arrow(pa.Table.from_pylist(self.rows, schema=self.schema)),
                       ray.data.from_items([])):
                with self.assertRaisesRegex(ValueError, "target table does not exist"):
                    writer.export(ds)
        self.assertFalse(os.path.exists(self.anchor))
        self.assertFalse(os.path.exists(os.path.dirname(self.anchor)))


if __name__ == '__main__':
    unittest.main()