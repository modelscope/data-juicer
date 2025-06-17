import os
import tempfile
import unittest
from unittest.mock import patch
from data_juicer.core.sandbox.env_manager import CondaEnv, VirtualEnv


class TestCondaEnv(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.env_name = "test_conda_env"
        self.config_path = os.path.join(self.test_dir.name, "env_config.yaml")
        with open(self.config_path, 'w') as f:
            f.write(f"name: {self.env_name}\n")
        self.requirements_file_path = os.path.join(self.test_dir.name, "requirements.txt")
        with open(self.requirements_file_path, 'w') as f:
            f.write("numpy\npandas\n")
        self.lib_home = self.test_dir.name

    @patch("data_juicer.core.sandbox.env_manager.shutil.which", return_value="/usr/bin/conda")
    def test_initialization_without_any_info(self, mock_which):
        with self.assertRaises(ValueError):
            _ = CondaEnv()

    @patch("data_juicer.core.sandbox.env_manager.shutil.which", return_value="/usr/bin/conda")
    def test_initialization_with_name(self, mock_which):
        env = CondaEnv(env_name=self.env_name)
        self.assertEqual(env.env_name, self.env_name)

    @patch("data_juicer.core.sandbox.env_manager.shutil.which", return_value="/usr/bin/conda")
    def test_initialization_with_config_path(self, mock_which):
        env = CondaEnv(env_config_path=self.config_path)
        self.assertEqual(env.env_name, self.env_name)

    @patch("data_juicer.core.sandbox.env_manager.shutil.which", return_value=None)
    def test_check_availability_failed(self, mock_which):
        with self.assertRaises(ValueError):
            CondaEnv(env_name="dummy").check_availability()

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_create_env_but_exists(self, mock_run):
        mock_run.return_value.stdout = '{"envs": ["/mock/envs/%s"]}' % self.env_name
        env = CondaEnv(env_name=self.env_name)
        self.assertTrue(env.exists())
        # called in exists
        mock_run.assert_called_once()
        env.create()
        # called in exists again and no new calling on subprocess.run to create env
        self.assertEqual(mock_run.call_count, 2)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_create_new_env_from_name(self, mock_run):
        mock_run.return_value.returncode = 0
        env = CondaEnv(env_name=self.env_name, env_py_version="3.9")
        env.create()
        # called in exists and called once when creating this new env
        self.assertEqual(mock_run.call_count, 2)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_create_new_env_from_config_path(self, mock_run):
        mock_run.return_value.returncode = 0
        env = CondaEnv(env_config_path=self.config_path)
        env.create()
        # called in exists and called once when creating this new env
        self.assertEqual(mock_run.call_count, 2)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_create_new_env_failed(self, mock_run):
        mock_run.return_value.returncode = 1
        with self.assertRaises(ValueError):
            env = CondaEnv(env_config_path=self.config_path)
            env.create()

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_exists_false(self, mock_run):
        mock_run.return_value.stdout = '{"envs": []}'
        env = CondaEnv(env_name=self.env_name)
        self.assertFalse(env.exists())

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_requirements_file(self, mock_run):
        mock_run.return_value.returncode = 0
        env = CondaEnv(env_name=self.env_name)
        env.install_py_deps(self.requirements_file_path)
        self.assertTrue(mock_run.called)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_lib_home(self, mock_run):
        mock_run.return_value.returncode = 0
        env = CondaEnv(env_name=self.env_name)
        env.install_py_deps(self.lib_home)
        self.assertTrue(mock_run.called)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_non_existing_path(self, mock_run):
        with self.assertRaises(FileNotFoundError):
            mock_run.return_value.returncode = 0
            env = CondaEnv(env_name=self.env_name)
            env.install_py_deps("/non/existing/path")
        self.assertFalse(mock_run.called)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_list(self, mock_run):
        mock_run.return_value.returncode = 0
        env = CondaEnv(env_name=self.env_name)
        env.install_py_deps(["numpy", "pandas"])
        self.assertTrue(mock_run.called)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_invalid(self, mock_run):
        with self.assertRaises(TypeError):
            mock_run.return_value.returncode = 0
            env = CondaEnv(env_name=self.env_name)
            env.install_py_deps({"invalid_input": "invalid_val"})
        self.assertFalse(mock_run.called)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_failed(self, mock_run):
        with self.assertRaises(RuntimeError):
            mock_run.return_value.returncode = 1
            env = CondaEnv(env_name=self.env_name)
            env.install_py_deps(["numpy", "pandas"])
        self.assertTrue(mock_run.called)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_run_cmd(self, mock_run):
        mock_run.return_value.returncode = 0
        env = CondaEnv(env_name=self.env_name)
        env.run_cmd("echo hello")
        self.assertTrue(mock_run.called)


class TestVirtualEnv(unittest.TestCase):
    def setUp(self):
        self.env_name = 'test_virtual_env'
        self.test_dir = tempfile.TemporaryDirectory()
        self.requirements_file_path = os.path.join(self.test_dir.name, "requirements.txt")
        with open(self.requirements_file_path, 'w') as f:
            f.write("numpy\npandas\n")
        self.lib_home = self.test_dir.name

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_initialization(self, mock_run):
        mock_run.return_value.returncode = 0
        env = VirtualEnv(env_manager="venv", env_name=self.env_name)
        self.assertEqual(env.env_name, self.env_name)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_initialization_without_info(self, mock_run):
        with self.assertRaises(ValueError):
            _ = VirtualEnv()

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_create(self, mock_run):
        mock_run.return_value.returncode = 0
        env = VirtualEnv(env_name=self.env_name)
        env.create()
        self.assertTrue(mock_run.called)

    def test_exists_false(self):
        env = VirtualEnv(env_name=self.env_name)
        self.assertFalse(env.exists())

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_list(self, mock_run):
        mock_run.return_value.returncode = 0
        env = VirtualEnv(env_name=self.env_name)
        env.install_py_deps(["requests"])
        self.assertEqual(mock_run.call_count, 2)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_requirements_file(self, mock_run):
        mock_run.return_value.returncode = 0
        env = VirtualEnv(env_name=self.env_name)
        env.install_py_deps(self.requirements_file_path)
        self.assertEqual(mock_run.call_count, 2)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_lib_home(self, mock_run):
        mock_run.return_value.returncode = 0
        env = VirtualEnv(env_name=self.env_name)
        env.install_py_deps(self.lib_home)
        self.assertEqual(mock_run.call_count, 2)

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_from_non_existing_path(self, mock_run):
        with self.assertRaises(FileNotFoundError):
            mock_run.return_value.returncode = 0
            env = VirtualEnv(env_name=self.env_name)
            env.install_py_deps("/non/existing/path")

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_invalid(self, mock_run):
        with self.assertRaises(TypeError):
            mock_run.return_value.returncode = 0
            env = VirtualEnv(env_name=self.env_name)
            env.install_py_deps({"invalid_input": "invalid_val"})

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_install_deps_failed(self, mock_run):
        with self.assertRaises(RuntimeError):
            mock_run.return_value.returncode = 0
            env = VirtualEnv(env_name=self.env_name)
            mock_run.return_value.returncode = 1
            env.install_py_deps(["numpy", "pandas"])

    @patch("data_juicer.core.sandbox.env_manager.subprocess.run")
    def test_run_cmd(self, mock_run):
        mock_run.return_value.returncode = 0
        env = VirtualEnv(env_name=self.env_name)
        env.run_cmd("echo hello")
        self.assertEqual(mock_run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
