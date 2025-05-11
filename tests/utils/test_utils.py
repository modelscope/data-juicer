import os
import sys
import subprocess
import tempfile
import shutil
import git
from pathlib import Path

def install_with_uv_or_pip(package_spec, editable=False):
    """
    Install a package using uv if available, otherwise fall back to pip.
    
    Args:
        package_spec: The package specification to install
        editable: Whether to install in editable mode
    """
    try:
        # Try uv first
        cmd = [sys.executable, '-m', 'uv', 'pip', 'install']
        if editable:
            cmd.append('-e')
        # Split package_spec if it contains flags
        if package_spec.startswith('-'):
            cmd.extend(package_spec.split())
        else:
            cmd.append(package_spec)
        print(f"Installing with uv: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv not found or failed, falling back to pip...")
        # Fall back to pip
        cmd = [sys.executable, '-m', 'pip', 'install']
        if editable:
            cmd.append('-e')
        # Split package_spec if it contains flags
        if package_spec.startswith('-'):
            cmd.extend(package_spec.split())
        else:
            cmd.append(package_spec)
        print(f"Installing with pip: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        return True

def get_dependency_files(package_dir):
    """
    Check for common dependency specification files in the package directory.
    Returns a list of (file_path, install_method) tuples.
    """
    dependency_files = []
    
    # Check for requirements.txt
    req_txt = os.path.join(package_dir, 'requirements.txt')
    if os.path.exists(req_txt):
        dependency_files.append((req_txt, '-r'))
    
    # Check for setup.py
    setup_py = os.path.join(package_dir, 'setup.py')
    if os.path.exists(setup_py):
        dependency_files.append((package_dir, '-e'))
    
    # Check for pyproject.toml
    pyproject = os.path.join(package_dir, 'pyproject.toml')
    if os.path.exists(pyproject):
        dependency_files.append((package_dir, '-e'))
    
    return dependency_files

def install_ram_dependencies():
    """
    Install RAM and its dependencies properly.
    This function:
    1. Clones RAM repository
    2. Detects and installs dependencies using any available specification method
    3. Installs RAM in editable mode if not already handled
    """
    try:
        # Try importing ram to check if it's installed
        import ram
        print("RAM is already installed.")
        return
    except ImportError:
        print("Installing RAM...")
        
        # Create a temporary directory for cloning
        temp_dir = tempfile.mkdtemp()
        try:
            # Clone the repository
            repo_url = 'https://github.com/xinyu1205/recognize-anything.git'
            print(f"Cloning {repo_url}...")
            git.Repo.clone_from(repo_url, temp_dir)
            
            # Get all dependency specification files
            dependency_files = get_dependency_files(temp_dir)
            
            if not dependency_files:
                print("Warning: No dependency specification files found.")
            
            # Install dependencies using all available methods
            for file_path, install_method in dependency_files:
                print(f"Installing dependencies from {os.path.basename(file_path)}...")
                if install_method == '-r':
                    install_with_uv_or_pip(f'-r {file_path}')
                else:  # -e for setup.py or pyproject.toml
                    install_with_uv_or_pip(file_path, editable=True)
            
            # If no setup.py or pyproject.toml was found, install in editable mode
            if not any(method == '-e' for _, method in dependency_files):
                print("Installing RAM in editable mode...")
                install_with_uv_or_pip(temp_dir, editable=True)
            
            print("RAM installation completed successfully.")
        except Exception as e:
            print(f"Error installing RAM: {str(e)}")
            raise
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

def setup_test_environment():
    """
    Set up the test environment by installing all required dependencies.
    This should be called before running tests that require RAM.
    """
    install_ram_dependencies()

if __name__ == '__main__':
    # If this file is run directly, install dependencies
    setup_test_environment() 