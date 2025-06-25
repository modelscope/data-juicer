#!/usr/bin/env python3
"""
Utility script to generate uv.lock file.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import tomli
import tomli_w


def read_pyproject_toml():
    """Read and parse pyproject.toml file."""
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'
    if not pyproject_path.exists():
        raise FileNotFoundError(
            f'pyproject.toml not found at {pyproject_path}')
    with open(pyproject_path, 'rb') as f:
        return tomli.load(f)


def check_uv_installed():
    """Check if uv is installed and available in PATH."""
    try:
        subprocess.run(['uv', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            'uv is not installed or not in PATH. Please install it first: '
            'pip install uv')


def generate_uv_lock():
    """Generate uv.lock file."""
    # Check prerequisites
    check_uv_installed()

    # Read pyproject.toml
    pyproject = read_pyproject_toml()
    pyproject_path = Path(__file__).parent.parent / 'pyproject.toml'

    # Backup original pyproject.toml
    backup_path = pyproject_path.with_suffix('.toml.bak')
    shutil.copy2(pyproject_path, backup_path)

    try:
        # Write modified pyproject.toml
        toml_str = tomli_w.dumps(pyproject)
        with open(pyproject_path, 'w', encoding='utf-8') as f:
            f.write(toml_str)

        # Generate uv.lock using uv lock
        subprocess.run(['uv', 'lock'], check=True)
        print('Successfully generated uv.lock')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore original pyproject.toml
        shutil.copy2(backup_path, pyproject_path)
        backup_path.unlink()


def main():
    """Main entry point."""
    try:
        generate_uv_lock()
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
