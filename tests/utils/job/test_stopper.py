import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import psutil

from data_juicer.utils.job.stopper import JobStopper, main, stop_job
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class JobStopperTest(DataJuicerTestCaseBase):

    def _stopper(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, "job"))
        return JobStopper("job", base_dir=tmpdir)

    @TEST_TAG("standalone")
    def test_init_and_missing_job(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            self.assertEqual(stopper.job_id, "job")
            with self.assertRaises(FileNotFoundError):
                JobStopper("missing", base_dir=tmpdir)

    @TEST_TAG("standalone")
    def test_terminate_process_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            proc = MagicMock(pid=123)
            self.assertTrue(stopper.terminate_process_gracefully(proc, timeout=2))

        proc.terminate.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=2)
        proc.kill.assert_not_called()

    @TEST_TAG("standalone")
    def test_terminate_process_force_kills_after_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            proc = MagicMock(pid=123)
            proc.wait.side_effect = [psutil.TimeoutExpired(2), None]
            self.assertTrue(stopper.terminate_process_gracefully(proc, timeout=2))

        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_count, 2)

    @TEST_TAG("standalone")
    def test_terminate_process_handles_errors(self):
        cases = (
            (psutil.NoSuchProcess(123), True),
            (psutil.AccessDenied(123), False),
            (RuntimeError("boom"), False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            for error, expected in cases:
                with self.subTest(error=type(error).__name__):
                    proc = MagicMock(pid=123)
                    proc.terminate.side_effect = error
                    self.assertEqual(stopper.terminate_process_gracefully(proc), expected)

    @TEST_TAG("standalone")
    def test_cleanup_updates_summary_with_mocked_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            summary_path = os.path.join(tmpdir, "job", "job_summary.json")
            with open(summary_path, "w") as stream:
                json.dump({"status": "running", "start_time": 1}, stream)
            with patch("data_juicer.utils.job.stopper.time.time", return_value=9.5):
                stopper.cleanup_job_resources()
            with open(summary_path) as stream:
                summary = json.load(stream)

        self.assertEqual(summary["status"], "stopped")
        self.assertEqual(summary["stop_time"], 9.5)
        self.assertEqual(summary["stop_reason"], "manual_stop")

    @TEST_TAG("standalone")
    def test_cleanup_handles_empty_summary_and_write_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            stopper.cleanup_job_resources()
            stopper.job_utils.load_job_summary = MagicMock(return_value={"status": "running"})
            with patch("builtins.open", mock_open()) as opened:
                opened.side_effect = PermissionError("denied")
                stopper.cleanup_job_resources()

    @TEST_TAG("standalone")
    def test_stop_job_without_ids_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            stopper.job_utils.load_job_summary = MagicMock(return_value=None)
            stopper.job_utils.extract_process_thread_ids = MagicMock(
                return_value={"process_ids": set(), "thread_ids": set()}
            )
            stopper.cleanup_job_resources = MagicMock()
            result = stopper.stop_job()

        self.assertFalse(result["success"])
        self.assertEqual(result["errors"], ["No process or thread IDs found in event logs"])
        stopper.cleanup_job_resources.assert_called_once_with()

    @TEST_TAG("standalone")
    def test_stop_job_tracks_successes_and_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            good = MagicMock(pid=11)
            bad = MagicMock(pid=22)
            stopper.job_utils.load_job_summary = MagicMock(
                return_value={"status": "running", "start_time": 1}
            )
            stopper.job_utils.extract_process_thread_ids = MagicMock(
                return_value={"process_ids": {11, 22}, "thread_ids": {33}}
            )
            stopper.job_utils.find_processes_by_ids = MagicMock(return_value=[good, bad])
            stopper.job_utils.find_threads_by_ids = MagicMock(return_value=[object()])
            stopper.terminate_process_gracefully = MagicMock(side_effect=[True, False])
            stopper.cleanup_job_resources = MagicMock()
            result = stopper.stop_job(force=True, timeout=7)

        self.assertTrue(result["success"])
        self.assertEqual(result["processes_found"], 2)
        self.assertEqual(result["processes_terminated"], 1)
        self.assertEqual(result["threads_terminated"], 1)
        self.assertEqual(result["errors"], ["Failed to terminate process 22"])
        stopper.terminate_process_gracefully.assert_any_call(good, 7)

    @TEST_TAG("standalone")
    def test_stop_job_found_ids_but_no_running_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stopper = self._stopper(tmpdir)
            stopper.job_utils.load_job_summary = MagicMock(return_value={"status": "running"})
            stopper.job_utils.extract_process_thread_ids = MagicMock(
                return_value={"process_ids": {11}, "thread_ids": {33}}
            )
            stopper.job_utils.find_processes_by_ids = MagicMock(return_value=[])
            stopper.job_utils.find_threads_by_ids = MagicMock(return_value=[])
            stopper.cleanup_job_resources = MagicMock()
            result = stopper.stop_job()

        self.assertFalse(result["success"])
        self.assertEqual(result["processes_terminated"], 0)
        self.assertEqual(result["threads_terminated"], 0)


class StopperConvenienceAndCliTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_stop_job_convenience_function(self):
        expected = {"success": True}
        with patch("data_juicer.utils.job.stopper.JobStopper") as stopper_cls:
            stopper_cls.return_value.stop_job.return_value = expected
            result = stop_job("abc", "/tmp/base", force=True, timeout=4)

        stopper_cls.assert_called_once_with("abc", "/tmp/base")
        stopper_cls.return_value.stop_job.assert_called_once_with(force=True, timeout=4)
        self.assertIs(result, expected)

    @TEST_TAG("standalone")
    def test_main_lists_jobs_and_empty_state(self):
        jobs = [
            {"job_id": "done", "status": "completed", "processes": 0},
            {"job_id": "live", "status": "running", "processes": 2},
            {"job_id": "bad", "status": "failed", "processes": 0},
        ]
        for returned, expected in ((jobs, "done"), ([], "No DataJuicer jobs found")):
            with self.subTest(empty=not returned), patch("sys.argv", ["stopper", "--list"]), patch(
                "data_juicer.utils.job.stopper.list_running_jobs", return_value=returned
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    main()
                self.assertIn(expected, output.getvalue())

    @TEST_TAG("standalone")
    def test_main_requires_job_id(self):
        with patch("sys.argv", ["stopper"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)

    @TEST_TAG("standalone")
    def test_main_success_and_verbose(self):
        result = {"success": True, "processes_terminated": 2, "threads_terminated": 1, "errors": []}
        with patch(
            "sys.argv", ["stopper", "abc", "--base-dir", "/tmp/base", "--force", "--timeout", "4", "--verbose"]
        ), patch("data_juicer.utils.job.stopper.stop_job", return_value=result) as stop, patch(
            "data_juicer.utils.job.stopper.logger.remove"
        ) as remove, patch("data_juicer.utils.job.stopper.logger.add") as add:
            output = io.StringIO()
            with redirect_stdout(output):
                main()

        remove.assert_called_once_with()
        add.assert_called_once()
        stop.assert_called_once_with("abc", "/tmp/base", force=True, timeout=4)
        self.assertIn("Terminated 2 processes", output.getvalue())

    @TEST_TAG("standalone")
    def test_main_failure_exits(self):
        result = {"success": False, "processes_terminated": 0, "threads_terminated": 0, "errors": ["failed"]}
        with patch("sys.argv", ["stopper", "abc"]), patch(
            "data_juicer.utils.job.stopper.stop_job", return_value=result
        ):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
