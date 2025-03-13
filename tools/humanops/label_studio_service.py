#!/usr/bin/env python
# flake8: noqa: E501
"""
Utility script to manage Label Studio using Docker/pip for Data Juicer.
It's intended for localhost development/debug purposes.

This script:
1. Checks if Docker is installed and running
2. Pulls the Label Studio Docker image if needed
3. Starts a Label Studio container with proper configuration
4. Creates a test project with sample configuration
5. Outputs connection details for use in Data Juicer

Usage:
    python label_studio_service.py [--port PORT] [--data-dir PATH] [--create-test-project]
    python label_studio_service.py --status
    python label_studio_service.py --stop
    python label_studio_service.py --kill

Options:
    --port PORT                  Port to run Label Studio on (default: 7070)
    --data-dir PATH              Directory to store Label Studio data (default: ./label_studio_data)
    --create-test-project        Create a test project with sample configuration
    --username USERNAME          Admin username for Label Studio (default: admin@example.com)
    --password PASSWORD          Admin password for Label Studio (default: admin)
    --api-token API_TOKEN        Manually specify the API token (skip automatic retrieval)
    --image IMAGE                Label Studio Docker image (default: heartexlabs/label-studio:latest)
    --container-name NAME        Name for Docker container (default: data-juicer-label-studio)
    --network NETWORK            Docker network to connect to (default: None)
    --stop                       Stop and remove the Label Studio container
    --kill                       Force kill and remove the Label Studio container
    --status                     Check if Label Studio container is running
"""

import argparse
import atexit
import fcntl
import json
import logging
import os
import select
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler

import requests

# Global logger
logger = None


def setup_logging(data_dir):
    """Setup logging to both file and console"""
    # Create logs directory
    logs_dir = os.path.join(os.path.abspath(data_dir), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(logs_dir, 'label_studio_service.log')

    # Create logger
    logger = logging.getLogger('label_studio_service')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create file handler with rotation (10MB max size, keep 3 backup files)
    file_handler = RotatingFileHandler(log_file,
                                       maxBytes=10 * 1024 * 1024,
                                       backupCount=3)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Start a Label Studio Docker container')
    parser.add_argument('--port',
                        type=int,
                        default=7070,
                        help='Port to run Label Studio on (default: 7070)')
    parser.add_argument('--data-dir',
                        default='./label_studio_data',
                        help='Directory to store Label Studio data')
    parser.add_argument(
        '--username',
        help='Username for Label Studio (default: will prompt if not provided)'
    )
    parser.add_argument(
        '--password',
        help='Password for Label Studio (default: will prompt if not provided)'
    )
    parser.add_argument('--api-token',
                        help='Specify a fixed API token for the admin account')
    parser.add_argument(
        '--image',
        default='heartexlabs/label-studio:latest',
        help='Docker image to use (default: heartexlabs/label-studio:latest)')
    parser.add_argument('--container-name',
                        default='data-juicer-label-studio',
                        help='Name for the Docker container')
    parser.add_argument('--network', help='Docker network to connect to')
    parser.add_argument('--status',
                        action='store_true',
                        help='Check if Label Studio container is running')
    parser.add_argument('--stop',
                        action='store_true',
                        help='Stop the Label Studio container')
    parser.add_argument('--kill',
                        action='store_true',
                        help='Force kill the Label Studio container')
    parser.add_argument('--create-test-project',
                        action='store_true',
                        help='Create a test project in Label Studio')
    return parser.parse_args()


def check_docker_installed():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', 'info'],
                                capture_output=True,
                                text=True,
                                check=False)
        if result.returncode == 0:
            logger.info('Docker is installed and running.')
            return True
        else:
            logger.error(
                'Docker is installed but not running or has permission issues.'
            )
            logger.error(result.stderr)
            return False
    except FileNotFoundError:
        logger.error('Docker is not installed or not in PATH.')
        return False


def pull_docker_image(image):
    """Pull the Label Studio Docker image"""
    logger.info(f'Pulling Docker image: {image}...')
    result = subprocess.run(['docker', 'pull', image], check=False)
    return result.returncode == 0


def check_container_exists(container_name):
    """Check if a container with the given name exists"""
    result = subprocess.run([
        'docker', 'ps', '-a', '--filter', f'name={container_name}', '--format',
        '{{.Names}}'
    ],
                            capture_output=True,
                            text=True,
                            check=False)
    return container_name in result.stdout.strip().split('\n')


def check_container_running(container_name):
    """Check if a container is currently running"""
    result = subprocess.run([
        'docker', 'ps', '--filter', f'name={container_name}', '--format',
        '{{.Names}}'
    ],
                            capture_output=True,
                            text=True,
                            check=False)
    return container_name in result.stdout.strip().split('\n')


def stop_container(container_name):
    """Stop and remove the Label Studio container"""
    if check_container_exists(container_name):
        logger.info(f'Stopping container: {container_name}...')
        subprocess.run(['docker', 'stop', container_name], check=False)
        logger.info(f'Removing container: {container_name}...')
        subprocess.run(['docker', 'rm', container_name], check=False)
        logger.info(f'Container {container_name} stopped and removed.')
        return True
    else:
        logger.info(f'Container {container_name} does not exist.')
        return False


def kill_container(container_name):
    """Force kill the Label Studio container"""
    if check_container_running(container_name):
        logger.info(f'Force killing container: {container_name}...')
        subprocess.run(['docker', 'kill', container_name], check=False)
        logger.info(f'Removing container: {container_name}...')
        subprocess.run(['docker', 'rm', '-f', container_name], check=False)
        logger.info(f'Container {container_name} killed and removed.')
        return True
    elif check_container_exists(container_name):
        logger.info(
            f'Container {container_name} exists but is not running. Removing...'
        )
        subprocess.run(['docker', 'rm', '-f', container_name], check=False)
        logger.info(f'Container {container_name} removed.')
        return True
    else:
        logger.info(f'Container {container_name} does not exist.')
        return False


def start_label_studio_container(port,
                                 data_dir,
                                 username,
                                 password,
                                 image,
                                 container_name,
                                 network=None,
                                 predefined_token=None):
    """Start a Label Studio Docker container"""
    # Ensure data directory exists
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Stop existing container if it exists
    if check_container_exists(container_name):
        stop_container(container_name)

    # Generate a fixed token if not provided
    if not predefined_token:
        import uuid
        predefined_token = str(uuid.uuid4())
        logger.info(f'Generated fixed API token: {predefined_token}')

    # Build docker run command
    cmd = [
        'docker',
        'run',
        '-d',
        '--name',
        container_name,
        '-p',
        f'{port}:8080',
        '-v',
        f'{data_dir}:/label-studio/data',
        '-e',
        f'LABEL_STUDIO_USERNAME={username}',
        '-e',
        f'LABEL_STUDIO_PASSWORD={password}',
        '-e',
        'LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true',
        '-e',
        f'LABEL_STUDIO_USER_TOKEN={predefined_token}'  # Set fixed token
    ]

    # Add network if specified
    if network:
        cmd.extend(['--network', network])

    # Add image name
    cmd.append(image)

    logger.info(f'Starting Label Studio container on port {port}...')
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error(f'Failed to start container: {result.stderr}')
        return None, None

    container_id = result.stdout.strip()
    logger.info(f'Container started with ID: {container_id}')

    # Register cleanup function to stop the container on exit
    atexit.register(lambda: stop_container(container_name))

    # Wait for server to start with better log monitoring
    server_url = f'http://localhost:{port}'
    max_wait_time = 600  # 10 minutes total wait time
    start_time = time.time()
    check_interval = 5  # Check every 5 seconds

    logger.info(
        'Waiting for Label Studio to initialize (this may take several minutes on first run)...'
    )
    logger.info('Monitoring container logs for startup progress...')

    # Initialization stages to look for in logs
    init_stages = {
        'database_init': 'Initializing database',
        'migrations': 'Applying migrations',
        'collecting_static': 'Collecting static files',
        'starting_server': 'Starting server',
        'server_running': 'Server is running'
    }

    completed_stages = set()
    last_log_position = 0

    while time.time() - start_time < max_wait_time:
        # Check if container is still running
        if not check_container_running(container_name):
            logger.info('Container stopped unexpectedly. Checking logs...')
            log_cmd = ['docker', 'logs', container_name]
            log_result = subprocess.run(log_cmd,
                                        capture_output=True,
                                        text=True,
                                        check=False)
            if log_result.returncode == 0:
                logger.info('Container logs:')
                logger.info(
                    log_result.stdout[-2000:])  # Show last 2000 chars of logs
            return None, None

        # Try health endpoint
        try:
            response = requests.get(f'{server_url}/health', timeout=5)
            if response.status_code == 200:
                logger.info(f'Label Studio is running at {server_url}')
                return server_url, predefined_token
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout):
            pass  # Expected during startup

        # Check logs for progress
        log_cmd = ['docker', 'logs', container_name]
        log_result = subprocess.run(log_cmd,
                                    capture_output=True,
                                    text=True,
                                    check=False)

        if log_result.returncode == 0:
            logs = log_result.stdout

            # Only process new log content
            new_logs = logs[last_log_position:]
            last_log_position = len(logs)

            if new_logs:
                # Print each log line individually for better readability
                for line in new_logs.strip().split('\n'):
                    if line.strip():
                        logger.info(f'CONTAINER: {line.strip()}')

                # Check for initialization stages
                for stage, marker in init_stages.items():
                    if stage not in completed_stages and marker in logs:
                        completed_stages.add(stage)
                        logger.info(f'✓ Detected stage: {marker}')

                # Special case: look for database initialization completion
                if 'database_init' in completed_stages and 'migrations' not in completed_stages:
                    if "Migrations for 'auth':" in logs:
                        completed_stages.add('migrations')
                        logger.info(f'\n✓ Detected stage: Starting migrations')

                # Special case: look for server startup
                if 'Starting development server at' in logs:
                    logger.info(f'\n✓ Detected stage: Server starting up')
                    # Give the server a moment to fully initialize
                    time.sleep(5)
                    try:
                        response = requests.get(f'{server_url}/health',
                                                timeout=5)
                        if response.status_code == 200:
                            logger.info(
                                f'Label Studio is running at {server_url}')
                            return server_url, predefined_token
                    except (requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout):
                        logger.info(
                            'Server starting but health endpoint not ready yet...'
                        )

        # Print periodic status update
        elapsed = int(time.time() - start_time)
        remaining = max_wait_time - elapsed
        if elapsed % 10 == 0:  # Status update every 10 seconds
            logger.info(
                f'\nStill waiting for Label Studio to initialize... ({elapsed}s elapsed, {remaining}s remaining)'
            )
            logger.info(
                f'Completed stages: {len(completed_stages)}/{len(init_stages)}'
            )

        time.sleep(check_interval)

    logger.info(
        f'Timeout after waiting {max_wait_time} seconds for Label Studio to start.'
    )
    logger.info('Final container logs:')
    log_cmd = ['docker', 'logs', '--tail', '100', container_name]
    log_result = subprocess.run(log_cmd,
                                capture_output=True,
                                text=True,
                                check=False)
    if log_result.returncode == 0:
        logger.info(log_result.stdout)

    logger.info(
        "Container is still running but Label Studio didn't respond in time.")
    logger.info('You may try accessing it manually at: ' + server_url)
    logger.info('Or stop the container with: docker stop ' + container_name)

    # Don't automatically stop the container - it might still be initializing
    return None, None


def get_api_token(server_url, username, password):
    """Get API token for Label Studio"""
    try:
        # Create a session to maintain cookies
        session = requests.Session()

        # Wait for server to be fully ready
        logger.info('Waiting for Label Studio to fully initialize...')
        time.sleep(20)  # Give more time for all services to start

        # First get the login page to establish session and get CSRF token
        logger.info(f'Getting login page from {server_url}/user/login')
        login_page = session.get(f'{server_url}/user/login', timeout=10)

        # Extract CSRF token from cookies or response
        csrf_token = None
        if 'csrftoken' in session.cookies:
            csrf_token = session.cookies['csrftoken']

        # Prepare login data - Label Studio expects form data, not JSON
        login_data = {
            'email': username,
            'password': password,
        }

        # Add CSRF token if available
        headers = {
            'Referer': f'{server_url}/user/login',
        }
        if csrf_token:
            headers['X-CSRFToken'] = csrf_token
            login_data['csrfmiddlewaretoken'] = csrf_token
            logger.info(f'Using CSRF token: {csrf_token[:10]}...')

        # Attempt login
        logger.info(f'Logging in as {username}...')
        login_response = session.post(
            f'{server_url}/user/login',
            data=login_data,  # Use form data instead of JSON
            headers=headers,
            timeout=10)

        if login_response.status_code != 200:
            logger.info(
                f'Login failed with status code: {login_response.status_code}')
            logger.info(f'Response: {login_response.text[:500]}')
            return None

        logger.info('Login successful!')

        # After successful login, get the user profile page which contains the token
        logger.info('Fetching user profile page...')
        profile_response = session.get(f'{server_url}/user/account',
                                       timeout=10)

        if profile_response.status_code != 200:
            logger.info(
                f'Failed to get profile page: {profile_response.status_code}')
            return None

        # Extract token from the profile page HTML
        import re
        token_match = re.search(r'id="access_token"[^>]*value="([^"]+)"',
                                profile_response.text)
        if token_match:
            token = token_match.group(1)
            logger.info(f'Successfully extracted API token from profile page')
            return token

        # If token not found in profile page, try API endpoint
        logger.info('Token not found in profile page, trying API endpoint...')
        token_response = session.get(f'{server_url}/api/current-user/token',
                                     timeout=10)

        if token_response.status_code == 200:
            try:
                token_data = token_response.json()
                if 'token' in token_data:
                    logger.info(
                        'Successfully obtained API token from endpoint')
                    return token_data['token']
            except:
                pass

        logger.error('Failed to automatically retrieve API token')
        return None

    except Exception as e:
        logger.error(f'Error getting API token: {e}')
        import traceback
        traceback.print_exc()
        return None


def create_test_project(server_url, api_token):
    """Create a test project in Label Studio"""
    project_url = f'{server_url}/api/projects'
    headers = {
        'Authorization': f'Token {api_token}',
        'Content-Type': 'application/json'
    }

    # Simple text classification project
    project_data = {
        'title':
        'Test Classification Project',
        'description':
        'A test project for text classification',
        'label_config':
        """
<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text" choice="single">
    <Choice value="Positive"/>
    <Choice value="Neutral"/>
    <Choice value="Negative"/>
  </Choices>
</View>
"""
    }

    try:
        response = requests.post(project_url,
                                 json=project_data,
                                 headers=headers)
        if response.status_code == 201:
            project = response.json()
            logger.info(f"Created test project: {project['title']} "
                        f"(ID: {project['id']})")

            # Create a sample task
            task_url = f"{server_url}/api/projects/{project['id']}/import"
            task_data = [{
                'data': {
                    'text':
                    'This is a sample text for annotation. '
                    'Please classify the sentiment.'
                }
            }]

            task_response = requests.post(task_url,
                                          json=task_data,
                                          headers=headers)
            if task_response.status_code == 201:
                logger.info('Added sample task to the project')
            else:
                logger.error(
                    f'Failed to add sample task: {task_response.text}')

            return project['id']
        else:
            logger.error(f'Failed to create test project: {response.text}')
            return None

    except Exception as e:
        logger.error(f'Error creating test project: {e}')
        return None


def load_connection_info():
    """Load connection information from file"""
    connection_file = 'label_studio_localhost_connection.json'
    if os.path.exists(connection_file):
        try:
            with open(connection_file, 'r') as f:
                connection_info = json.load(f)
                logger.info(
                    f'Loaded existing connection info from {connection_file}')
                return connection_info
        except Exception as e:
            logger.error(f'Error loading connection info: {e}')
    return None


def check_pip_installed():
    """Check if pip is installed"""
    try:
        subprocess.run(['pip', '--version'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=False)
        return True
    except FileNotFoundError:
        return False


def install_label_studio_pip():
    """Install Label Studio using pip"""
    logger.info('Installing Label Studio using pip...')
    try:
        # Create a virtual environment
        venv_dir = os.path.abspath('label_studio_venv')
        if not os.path.exists(venv_dir):
            logger.info(f'Creating virtual environment at {venv_dir}')
            subprocess.run([sys.executable, '-m', 'venv', venv_dir],
                           check=True)

        # Determine the pip and python executables in the venv
        if os.name == 'nt':  # Windows
            pip_cmd = os.path.join(venv_dir, 'Scripts', 'pip')
            python_cmd = os.path.join(venv_dir, 'Scripts', 'python')
        else:  # Unix/Linux/Mac
            pip_cmd = os.path.join(venv_dir, 'bin', 'pip')
            python_cmd = os.path.join(venv_dir, 'bin', 'python')

        # Upgrade pip
        logger.info('Upgrading pip...')
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)

        # Install Label Studio
        logger.info('Installing Label Studio...')
        subprocess.run([pip_cmd, 'install', 'label-studio'], check=True)

        logger.info('Label Studio installed successfully!')
        return venv_dir, python_cmd
    except subprocess.CalledProcessError as e:
        logger.error(f'Error installing Label Studio: {e}')
        return None, None


def start_label_studio_pip(data_dir,
                           username,
                           password,
                           port,
                           predefined_token=None):
    """Start Label Studio using pip installation"""
    venv_dir, python_cmd = install_label_studio_pip()
    if not venv_dir or not python_cmd:
        logger.error('Failed to install Label Studio.')
        return None, None

    # Ensure data directory exists
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Generate a fixed token if not provided
    if not predefined_token:
        import uuid
        predefined_token = str(uuid.uuid4())
        logger.info(f'Generated fixed API token: {predefined_token}')

    # Set environment variables
    env = os.environ.copy()
    env['LABEL_STUDIO_USERNAME'] = username
    env['LABEL_STUDIO_PASSWORD'] = password
    env['LABEL_STUDIO_USER_TOKEN'] = predefined_token
    env['LABEL_STUDIO_BASE_DATA_DIR'] = data_dir
    env['LABEL_STUDIO_PORT'] = str(port)

    # Start Label Studio
    logger.info(f'Starting Label Studio on port {port}...')
    try:
        # Use Popen to start the process in the background
        process = subprocess.Popen(
            [
                python_cmd, '-m', 'label_studio.server', 'start',
                '--no-browser', f'--port={port}'
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Register cleanup function to terminate the process on exit
        atexit.register(lambda: process.terminate())

        # Monitor the output for startup completion
        server_url = f'http://localhost:{port}'
        max_wait_time = 300  # 5 minutes
        start_time = time.time()

        logger.info('Waiting for Label Studio to initialize...')

        # Make stdout non-blocking
        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        # Initialize variables for tracking server status
        server_started = False

        # Process output in real-time
        while time.time() - start_time < max_wait_time:
            # Check if process is still running
            if process.poll() is not None:
                logger.error('Label Studio process exited unexpectedly.')
                return None, None

            # Use select to check if there's data to read without blocking
            ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)

            if ready_to_read:
                # Read all available output
                output = process.stdout.read()
                if output:
                    # Process each line
                    for line in output.splitlines():
                        if line.strip():
                            logger.info(f'LABEL_STUDIO: {line.strip()}')

                            # Check for successful startup message
                            if 'Label Studio is up and running' in line:
                                logger.info(
                                    f'Label Studio is running at {server_url}')
                                server_started = True

            # Check if the server is responding
            try:
                response = requests.get(f'{server_url}/health', timeout=2)
                if response.status_code == 200:
                    logger.info(f'Label Studio is running at {server_url}')
                    return server_url, predefined_token
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout):
                pass  # Expected during startup

            # If we've detected server startup in the logs, give it a moment to fully initialize
            if server_started:
                time.sleep(2)
                try:
                    response = requests.get(f'{server_url}/health', timeout=2)
                    if response.status_code == 200:
                        logger.info(f'Label Studio is running at {server_url}')
                        return server_url, predefined_token
                except (requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout):
                    # If health check fails, continue monitoring
                    server_started = False

        logger.error('Timed out waiting for Label Studio to start.')
        return None, None

    except Exception as e:
        logger.error(f'Error starting Label Studio: {e}')
        return None, None


def main():
    args = parse_args()

    # Initialize logging
    global logger
    logger = setup_logging(args.data_dir)

    logger.info('Label Studio Service starting')

    # Handle status request
    if args.status:
        if check_container_running(args.container_name):
            logger.info(f'Container {args.container_name} is running.')
            return 0
        elif check_container_exists(args.container_name):
            logger.info(
                f'Container {args.container_name} exists but is not running.')
            return 1
        else:
            logger.info(f'Container {args.container_name} does not exist.')
            return 1

    # Handle stop request
    if args.stop:
        stop_container(args.container_name)
        return 0

    # Handle kill request
    if args.kill:
        kill_container(args.container_name)
        return 0

    # Load existing connection info if available
    existing_connection = load_connection_info()

    # Initialize variables with values from connection file
    api_token = None
    project_id = None
    username_from_file = None
    password_from_file = None

    if existing_connection:
        if 'api_token' in existing_connection:
            api_token = existing_connection['api_token']

        if 'username' in existing_connection:
            username_from_file = existing_connection['username']

        if 'password' in existing_connection:
            password_from_file = existing_connection['password']

        if 'project_id' in existing_connection:
            project_id = existing_connection['project_id']

    # Override with command line arguments if provided
    if args.api_token:
        api_token = args.api_token
        logger.info(f'Overriding API token with command line argument')

    # Use username from command line or connection file, or prompt if neither is available
    if args.username:
        logger.info(f'Using username from command line: {args.username}')
    elif username_from_file:
        args.username = username_from_file
    else:
        import getpass
        logger.info('Label Studio requires an admin username.')
        args.username = input('Enter admin email: ').strip()
        if not args.username:
            args.username = 'admin@example.com'
            logger.info(f'Using default username: {args.username}')

    # Use password from command line or connection file, or prompt if neither is available
    if args.password:
        logger.info('Using password from command line')
    elif password_from_file:
        args.password = password_from_file
    else:
        import getpass
        logger.info('Label Studio requires an admin password.')
        args.password = getpass.getpass('Enter admin password: ').strip()
        if not args.password:
            import random
            import string

            # Generate a random password if none provided
            args.password = ''.join(
                random.choices(string.ascii_letters + string.digits, k=12))
            logger.info(f'Using generated password: {args.password}')

    # Validate email format
    import re
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    if not email_pattern.match(args.username):
        logger.info(
            f"Warning: '{args.username}' doesn't appear to be a valid email address."
        )
        logger.info(
            'Label Studio requires a valid email format for the username.')
        use_anyway = input('Use this username anyway? (y/n): ').strip().lower()
        if use_anyway != 'y':
            logger.info(
                'Please restart with a valid email address as the username.')
            return 1

    # Check if Docker is installed and running
    docker_available = check_docker_installed()

    # Start Label Studio
    server_url = None
    if docker_available:
        logger.info('Docker is available, using Docker to run Label Studio.')

        # Pull the Docker image
        if not pull_docker_image(args.image):
            logger.error(f'Failed to pull Docker image: {args.image}')
            logger.info('Falling back to pip installation...')
            docker_available = False

    if docker_available:
        # Start Label Studio container with predefined token
        server_url, api_token = start_label_studio_container(
            args.port,
            args.data_dir,
            args.username,
            args.password,
            args.image,
            args.container_name,
            args.network,
            api_token  # Use token from connection file or command line
        )
    else:
        logger.info(
            'Docker is not available, using pip to install and run Label Studio.'
        )
        if not check_pip_installed():
            logger.error('Error: Neither Docker nor pip is available.')
            logger.info('Please install either Docker or pip and try again.')
            return 1

        # Start Label Studio using pip
        server_url, api_token = start_label_studio_pip(
            args.data_dir,
            args.username,
            args.password,
            args.port,
            api_token  # Use token from connection file or command line
        )

    if not server_url:
        logger.error('Failed to start Label Studio. Exiting.')
        return 1

    # Create test project if requested
    if args.create_test_project and api_token:
        project_id = create_test_project(server_url, api_token)
        logger.info(f'Created test project with ID: {project_id}')

    # Display connection information but don't save it
    if api_token:
        connection_info = {
            'api_url': server_url,
            'api_key': api_token,
            'project_id': project_id
        }
        logger.info('\nConnection information:')
        logger.info(json.dumps(connection_info, indent=2))
        logger.info(
            '\nNote: Connection information is not being saved automatically.')
        logger.info(
            "If you need to save this information, you can create a 'label_studio_localhost_connection.json' file manually."
        )
    else:
        logger.info('\nNo API token available.')
        logger.info(
            f'You can still access Label Studio manually at {server_url}')

    logger.info('\nLabel Studio is running. Press Ctrl+C to stop.')
    logger.info(f'Access the web interface at: {server_url}')
    logger.info(f'Login with:')
    logger.info(f'  Username: {args.username}')
    logger.info(f'  Password: {args.password}')
    if api_token:
        logger.info(f'  API Token: {api_token}')

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('\nStopping Label Studio...')
        if docker_available:
            stop_container(args.container_name)
            logger.info('Label Studio container stopped.')
        else:
            logger.info('Label Studio process terminated.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
