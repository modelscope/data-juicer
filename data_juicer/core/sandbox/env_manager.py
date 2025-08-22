import json
import os.path
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import List, Union

import yaml
from loguru import logger


class Env(ABC):
    @abstractmethod
    def create(self):
        """
        Create an environment.
        """
        raise NotImplementedError("This method must be implemented in subclass.")

    @abstractmethod
    def check_availability(self):
        """
        Check the availability of the environment manager.
        """
        raise NotImplementedError("This method must be implemented in subclass.")

    @abstractmethod
    def exists(self):
        """
        Check if an environment exists.
        """
        raise NotImplementedError("This method must be implemented in subclass.")

    @abstractmethod
    def install_py_deps(self):
        """
        Install Python dependencies.
        """
        raise NotImplementedError("This method must be implemented in subclass.")

    @abstractmethod
    def run_cmd(self):
        """
        Run a command in this environment.
        """
        raise NotImplementedError("This method must be implemented in subclass.")


class CondaEnv(Env):
    """
    Conda environment.
    """

    SUPPORTED_MANAGERS = {"conda", "mamba"}

    def __init__(
        self, env_manager: str = "conda", env_config_path: str = None, env_name: str = None, env_py_version: str = None
    ):
        assert env_manager in self.SUPPORTED_MANAGERS
        self.env_manager = env_manager
        self.env_config_path = env_config_path
        self.env_name = env_name
        self.env_py_version = env_py_version or "3.10"

        self.check_availability()

        if self.env_config_path is None and self.env_name is None:
            raise ValueError("Either env_config_path or env_name must be specified.")

        if self.env_config_path is not None:
            config = yaml.safe_load(open(self.env_config_path, "r"))
            self.env_name = config["name"]

    def check_availability(self):
        """
        Check the availability of the environment manager.
        """
        if shutil.which(self.env_manager) is None:
            raise ValueError(f"{self.env_manager} is not available.")
        return True

    def create(self):
        """
        Create an environment.
        """
        if self.exists():
            logger.info(f"Environment {self.env_name} already exists.")
            return

        if self.env_config_path is not None:
            cmd = f"{self.env_manager} env create -f {self.env_config_path}"
        elif self.env_name is not None:
            cmd = f"{self.env_manager} create -n {self.env_name} -y"
            cmd += f" python={self.env_py_version}"
        else:
            raise ValueError("Either env_config_path or env_name must be specified.")

        # init the env
        cmd += f" && {self.env_manager} init"

        logger.info(f"Creating environment {self.env_name}...")
        res = subprocess.run(cmd, shell=True)
        if res.returncode == 0:
            logger.info(f"Environment [{self.env_name}] created successfully.")
            return True
        else:
            raise ValueError(f"Failed to create environment {self.env_name}.")

    def exists(self):
        """
        Check if an environment exists.
        """
        try:
            res = subprocess.run(
                [self.env_manager, "env", "list", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            envs = json.loads(res.stdout).get("envs", [])

            for env_path in envs:
                if os.path.basename(env_path) == self.env_name:
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to check environment existence: {e}.")
            return False

    def install_py_deps(self, deps: Union[str, List[str]]):
        """
        Install Python dependencies.
        Support 3 ways:
        1. given a requirements.txt file path.
        2. given a directory path to a library code base.
        3. given a list of deps.
        """
        if isinstance(deps, str):
            if os.path.exists(deps):
                if os.path.isdir(deps):
                    logger.info(f"Installing library code base [{deps}]...")
                    cmd = f"{self.env_manager} run -n {self.env_name} pip install -e {deps}"
                else:
                    logger.info(f"Installing from requirements file [{deps}]...")
                    cmd = f"{self.env_manager} run -n {self.env_name} pip install -r {deps}"
            else:
                raise FileNotFoundError(f"deps path [{deps}] does not exist.")
        elif isinstance(deps, list):
            logger.info(f"Installing Python dependencies [{deps}]...")
            cmd = f'{self.env_manager} run -n {self.env_name} pip install {" ".join(deps)}'
        else:
            raise TypeError(f"deps must be a string or a list, got {type(deps)}")

        res = subprocess.run(cmd, shell=True)
        if res.returncode == 0:
            logger.info("Python dependencies installed successfully.")
            return True
        else:
            raise RuntimeError("Failed to install Python dependencies.")

    def run_cmd(self, cmd: str, use_sys_stdio=False):
        """
        Run a command in this environment.
        """
        cmd = f"{self.env_manager} run -n {self.env_name} bash -c '{cmd}'"
        if use_sys_stdio:
            stdout, stderr = sys.stdout, sys.stderr
        else:
            stdout, stderr = None, None
        res = subprocess.run(cmd, shell=True, stdout=stdout, stderr=stderr)
        if res.returncode == 0:
            logger.debug(f"Command [{cmd}] executed successfully.")
            return True
        else:
            raise RuntimeError(f"Failed to execute command [{cmd}].")


class VirtualEnv(Env):
    """
    Conda environment.
    """

    SUPPORTED_MANAGERS = {"venv": "python -m venv", "virtualenv": "virtualenv", "uv": "uv venv"}

    def __init__(self, env_manager: str = "venv", env_name: str = None):
        assert env_manager in self.SUPPORTED_MANAGERS
        self.env_manager = self.SUPPORTED_MANAGERS[env_manager]
        self.env_name = env_name

        self.check_availability()

        if self.env_name is None:
            raise ValueError("env_name must be specified.")

    def check_availability(self):
        """
        Check the availability of the environment manager.
        """
        test_cmd = f"{self.env_manager} --help"
        res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        if res.returncode != 0:
            raise ValueError(f"{self.env_manager} is not available.")
        return True

    def create(self):
        """
        Create an environment.
        """
        if self.exists():
            logger.info(f"Environment {self.env_name} already exists.")
            return

        if self.env_name is not None:
            cmd = f"{self.env_manager} {self.env_name}"
        else:
            raise ValueError("Either env_config_path or env_name must be specified.")

        logger.info(f"Creating environment {self.env_name}...")
        res = subprocess.run(cmd, shell=True)
        if res.returncode == 0:
            logger.info(f"Environment [{self.env_name}] created successfully.")
            return True
        else:
            raise ValueError(f"Failed to create environment {self.env_name}.")

    def exists(self):
        """
        Check if an environment exists.
        """
        return (
            os.path.exists(self.env_name)
            and os.path.isdir(self.env_name)
            and os.path.exists(os.path.join(self.env_name, "bin", "activate"))
        )

    def install_py_deps(self, deps: Union[str, List[str]]):
        """
        Install Python dependencies.
        Support 3 ways:
        1. given a requirements.txt file path.
        2. given a directory path to a library code base.
        3. given a list of deps.
        """
        cmd = f"source {self.env_name}/bin/activate"
        if isinstance(deps, str):
            if os.path.exists(deps):
                if os.path.isdir(deps):
                    cmd += f" && pip install -e {deps}"
                else:
                    cmd += f" && pip install -r {deps}"
            else:
                raise FileNotFoundError(f"deps path [{deps}] does not exist.")
        elif isinstance(deps, list):
            cmd += f' && pip install {" ".join(deps)}'
        else:
            raise TypeError(f"deps must be a string or a list, got {type(deps)}")
        cmd += " && deactivate"

        res = subprocess.run(cmd, shell=True)
        if res.returncode == 0:
            logger.info("Python dependencies installed successfully.")
            return True
        else:
            raise RuntimeError("Failed to install Python dependencies.")

    def run_cmd(self, cmd: str):
        """
        Run a command in this environment.
        """
        cmd = f"source {self.env_name}/bin/activate && {cmd} && deactivate"
        res = subprocess.run(cmd, shell=True)
        if res.returncode == 0:
            logger.debug(f"Command [{cmd}] executed successfully.")
            return True
        else:
            raise RuntimeError(f"Failed to execute command [{cmd}].")


ALL_ENVS = {CondaEnv, VirtualEnv}

ENV_ROUTER = {}
for env_cls in ALL_ENVS:
    for env_name in env_cls.SUPPORTED_MANAGERS:
        ENV_ROUTER[env_name] = env_cls
