#!/usr/bin/env python
# flake8: noqa: E501
"""
Utility script to manage Label Studio using Docker for Data Juicer.

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
    --port PORT                  Port to run Label Studio on (default: 8080)
    --data-dir PATH              Directory to store Label Studio data (default: ./label_studio_data)
    --create-test-project        Create a test project with sample configuration
    --username USERNAME          Admin username for Label Studio (default: admin@example.com)
    --password PASSWORD          Admin password for Label Studio (default: admin)
    --image IMAGE                Label Studio Docker image (default: heartexlabs/label-studio:latest)
    --container-name NAME        Name for Docker container (default: data-juicer-label-studio)
    --network NETWORK            Docker network to connect to (default: None)
    --stop                       Stop and remove the Label Studio container
    --kill                       Force kill and remove the Label Studio container
    --status                     Check if Label Studio container is running
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description='Manage Label Studio using Docker')
    parser.add_argument('--port',
                        type=int,
                        default=8080,
                        help='Port to run Label Studio on (default: 8080)')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./label_studio_data',
        help='Directory to store data (default: ./label_studio_data)')
    parser.add_argument('--create-test-project',
                        action='store_true',
                        help='Create a test project with sample configuration')
    parser.add_argument(
        '--username',
        type=str,
        default='admin@example.com',
        help='Admin username for Label Studio (default: admin@example.com)')
    parser.add_argument(
        '--password',
        type=str,
        default='admin',
        help='Admin password for Label Studio (default: admin)')
    parser.add_argument(
        '--image',
        type=str,
        default='heartexlabs/label-studio:latest',
        help=
        'Label Studio Docker image (default: heartexlabs/label-studio:latest)')
    parser.add_argument(
        '--container-name',
        type=str,
        default='data-juicer-label-studio',
        help='Name for Docker container (default: data-juicer-label-studio)')
    parser.add_argument('--network',
                        type=str,
                        default=None,
                        help='Docker network to connect to (default: None)')
    parser.add_argument('--stop',
                        action='store_true',
                        help='Stop and remove the Label Studio container')
    parser.add_argument(
        '--kill',
        action='store_true',
        help='Force kill and remove the Label Studio container')
    parser.add_argument('--status',
                        action='store_true',
                        help='Check if Label Studio container is running')
    return parser.parse_args()


def check_docker_installed():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', 'info'],
                                capture_output=True,
                                text=True,
                                check=False)
        if result.returncode == 0:
            print('Docker is installed and running.')
            return True
        else:
            print(
                'Docker is installed but not running or has permission issues.'
            )
            print(result.stderr)
            return False
    except FileNotFoundError:
        print('Docker is not installed or not in PATH.')
        return False


def pull_docker_image(image):
    """Pull the Label Studio Docker image"""
    print(f'Pulling Docker image: {image}...')
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
        print(f'Stopping container: {container_name}...')
        subprocess.run(['docker', 'stop', container_name], check=False)
        print(f'Removing container: {container_name}...')
        subprocess.run(['docker', 'rm', container_name], check=False)
        print(f'Container {container_name} stopped and removed.')
        return True
    else:
        print(f'Container {container_name} does not exist.')
        return False


def kill_container(container_name):
    """Force kill the Label Studio container"""
    if check_container_running(container_name):
        print(f'Force killing container: {container_name}...')
        subprocess.run(['docker', 'kill', container_name], check=False)
        print(f'Removing container: {container_name}...')
        subprocess.run(['docker', 'rm', '-f', container_name], check=False)
        print(f'Container {container_name} killed and removed.')
        return True
    elif check_container_exists(container_name):
        print(
            f'Container {container_name} exists but is not running. Removing...'
        )
        subprocess.run(['docker', 'rm', '-f', container_name], check=False)
        print(f'Container {container_name} removed.')
        return True
    else:
        print(f'Container {container_name} does not exist.')
        return False


def start_label_studio_container(port,
                                 data_dir,
                                 username,
                                 password,
                                 image,
                                 container_name,
                                 network=None):
    """Start a Label Studio Docker container"""
    # Ensure data directory exists
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Stop existing container if it exists
    if check_container_exists(container_name):
        stop_container(container_name)

    # Build docker run command
    cmd = [
        'docker', 'run', '-d', '--name', container_name, '-p', f'{port}:8080',
        '-v', f'{data_dir}:/label-studio/data', '-e',
        f'LABEL_STUDIO_USERNAME={username}', '-e',
        f'LABEL_STUDIO_PASSWORD={password}', '-e',
        'LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true'
    ]

    # Add network if specified
    if network:
        cmd.extend(['--network', network])

    # Add image name
    cmd.append(image)

    print(f'Starting Label Studio container on port {port}...')
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(f'Failed to start container: {result.stderr}')
        return None

    container_id = result.stdout.strip()
    print(f'Container started with ID: {container_id}')

    # Register cleanup function to stop the container on exit
    atexit.register(lambda: stop_container(container_name))

    # Wait for server to start
    server_url = f'http://localhost:{port}'
    max_retries = 30
    retry_interval = 2

    print('Waiting for Label Studio to start...')
    for i in range(max_retries):
        try:
            response = requests.get(f'{server_url}/health')
            if response.status_code == 200:
                print(f'Label Studio is running at {server_url}')
                return server_url
        except requests.exceptions.ConnectionError:
            pass

        time.sleep(retry_interval)
        print(f'Waiting for server to start... ({i+1}/{max_retries})')

    print('Failed to start Label Studio server.')
    stop_container(container_name)
    return None


def get_api_token(server_url, username, password):
    """Get API token for Label Studio"""
    # First try to log in
    login_url = f'{server_url}/api/auth/login'
    login_data = {'email': username, 'password': password}

    try:
        response = requests.post(login_url, json=login_data)
        if response.status_code != 200:
            print(f'Failed to log in: {response.text}')
            return None

        # Get the token
        token_url = f'{server_url}/api/current-user/token'
        headers = {
            'Cookie':
            '; '.join([f'{k}={v}' for k, v in response.cookies.items()])
        }

        token_response = requests.get(token_url, headers=headers)
        if token_response.status_code != 200:
            print(f'Failed to get token: {token_response.text}')
            return None

        token = token_response.json().get('token')
        if token:
            print(f'Successfully obtained API token: {token}')
            return token
        else:
            print('Token not found in response')
            return None

    except Exception as e:
        print(f'Error getting API token: {e}')
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
            print(f"Created test project: {project['title']} "
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
                print('Added sample task to the project')
            else:
                print(f'Failed to add sample task: {task_response.text}')

            return project['id']
        else:
            print(f'Failed to create test project: {response.text}')
            return None

    except Exception as e:
        print(f'Error creating test project: {e}')
        return None


def save_connection_info(server_url,
                         api_token,
                         project_id,
                         output_file='label_studio_connection.json'):
    """Save connection information to a file for use in Data Juicer"""
    connection_info = {
        'api_url': server_url,
        'api_key': api_token,
        'project_id': project_id
    }

    with open(output_file, 'w') as f:
        json.dump(connection_info, f, indent=2)

    print(f'Connection information saved to {output_file}')
    print('\nTo use this in Data Juicer, add the following to your config:')
    print(f"""
annotation:
  label_studio:
    api_url: "{server_url}"
    api_key: "{api_token}"
    project_id: {project_id}
    project_name: "Test Classification Project"
    wait_for_annotations: false
    timeout: 3600
    poll_interval: 60
""")


def main():
    args = parse_args()

    # Handle status request
    if args.status:
        if check_container_running(args.container_name):
            print(f'Container {args.container_name} is running.')
            return 0
        elif check_container_exists(args.container_name):
            print(
                f'Container {args.container_name} exists but is not running.')
            return 1
        else:
            print(f'Container {args.container_name} does not exist.')
            return 1

    # Handle stop request
    if args.stop:
        stop_container(args.container_name)
        return 0

    # Handle kill request
    if args.kill:
        kill_container(args.container_name)
        return 0

    # Check if Docker is installed and running
    if not check_docker_installed():
        print('Docker is required to run Label Studio. '
              'Please install Docker and try again.')
        return 1

    # Pull the Docker image
    if not pull_docker_image(args.image):
        print(f'Failed to pull Docker image: {args.image}')
        return 1

    # Start Label Studio container
    server_url = start_label_studio_container(args.port, args.data_dir,
                                              args.username, args.password,
                                              args.image, args.container_name,
                                              args.network)

    if not server_url:
        print('Failed to start Label Studio container. Exiting.')
        return 1

    # Wait for server to be fully ready
    time.sleep(10)

    # Get API token
    api_token = get_api_token(server_url, args.username, args.password)
    if not api_token:
        print('Failed to get API token. Exiting.')
        return 1

    # Create test project if requested
    project_id = None
    if args.create_test_project:
        project_id = create_test_project(server_url, api_token)

    # Save connection information
    if api_token:
        save_connection_info(server_url, api_token, project_id)

    print('\nLabel Studio is running in Docker. Press Ctrl+C to stop.')

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nStopping Label Studio container...')
        stop_container(args.container_name)
        print('Label Studio container stopped.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
