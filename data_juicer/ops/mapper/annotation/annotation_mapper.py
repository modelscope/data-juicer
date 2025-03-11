import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from data_juicer.ops.mixins import EventDrivenMixin, NotificationMixin
from data_juicer.utils.constant import Fields

from ...base_op import Mapper

# Common annotation event types
ANNOTATION_EVENTS = {
    'TASK_CREATED': 'task_created',
    'TASK_ASSIGNED': 'task_assigned',
    'ANNOTATION_STARTED': 'annotation_started',
    'ANNOTATION_COMPLETED': 'annotation_completed',
    'ANNOTATION_REJECTED': 'annotation_rejected',
    'ANNOTATION_SKIPPED': 'annotation_skipped',
    'PROJECT_COMPLETED': 'project_completed',
    'ERROR_OCCURRED': 'error_occurred',
    'BATCH_CREATED': 'batch_created'
}


class BaseAnnotationMapper(EventDrivenMixin, NotificationMixin, Mapper, ABC):
    """Base class for annotation operations with event-driven capabilities"""

    _batched_op = True  # Mark this as a batched operator

    def __init__(self,
                 config_path: Optional[str] = None,
                 project_name: str = 'Annotation Project',
                 project_id: Optional[int] = None,
                 wait_for_annotations: bool = False,
                 timeout: int = 3600,
                 poll_interval: int = 60,
                 samples_per_task: int = 1,
                 max_tasks_per_batch: int = 100,
                 notification_config: Optional[Dict] = None,
                 notification_events: Optional[Dict[str, bool]] = None,
                 **kwargs):
        """Initialize the base annotation operation

        Args:
            config_path: Path to the configuration file
            project_name: Name of the project to create or use
            project_id: ID of existing project (if None, creates new project)
            wait_for_annotations: Whether to wait for annotations to complete
            timeout: Maximum time to wait for annotations in seconds
            poll_interval: Time between annotation status checks in seconds
            samples_per_task: Number of samples in each annotation task
            max_tasks_per_batch: Maximum number of tasks in a single batch
            notification_config: Configuration for notifications
            notification_events: Events that should trigger notifications
        """
        # Initialize with notification config
        super().__init__(notification_config=notification_config, **kwargs)

        # Store configuration
        self.project_name = project_name
        self.project_id = project_id
        self.wait_for_annotations = wait_for_annotations
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.samples_per_task = samples_per_task
        self.max_tasks_per_batch = max_tasks_per_batch

        # Configure which events should trigger notifications
        self.notification_events = notification_events or {
            'task_created': False,
            'batch_created': True,
            'annotation_completed': True,
            'project_completed': True,
            'error_occurred': True
        }

        # Track task IDs and sample mappings
        self.sample_to_task_id = {}  # Maps sample ID to task ID
        self.task_to_samples = defaultdict(
            list)  # Maps task ID to list of sample IDs
        self.processed_annotations = set()
        self.batch_counter = 0

        # Set up event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up default event handlers"""
        # Register event handlers with notification support
        self.register_event_handler(ANNOTATION_EVENTS['TASK_CREATED'],
                                    self._handle_task_created)
        self.register_event_handler(ANNOTATION_EVENTS['BATCH_CREATED'],
                                    self._handle_batch_created)
        self.register_event_handler(ANNOTATION_EVENTS['ANNOTATION_COMPLETED'],
                                    self._handle_annotation_completed)
        self.register_event_handler(ANNOTATION_EVENTS['ERROR_OCCURRED'],
                                    self._handle_error)

    def _handle_task_created(self, data):
        """Handle task created event with notification"""
        task_id = data.get('task_id')
        sample_ids = data.get('sample_ids', [])

        logger.info(f'Task {task_id} created with {len(sample_ids)} samples')

        # Send notification if configured
        if self.notification_events.get('task_created', False):
            self.send_notification(
                f'Task {task_id} created with {len(sample_ids)} '
                f'samples for annotation in project {self.project_name}',
                subject=f'New Annotation Task Created - {self.project_name}')

    def _handle_batch_created(self, data):
        """Handle batch created event with notification"""
        batch_id = data.get('batch_id')
        task_count = data.get('task_count', 0)
        sample_count = data.get('sample_count', 0)

        logger.info(
            f'Batch {batch_id} created with {task_count} tasks containing '
            f'{sample_count} samples')

        # Send notification if configured
        if self.notification_events.get('batch_created', False):
            self.send_notification(
                f'Batch {batch_id} created with {task_count} tasks containing '
                f'{sample_count} samples in project {self.project_name}',
                subject=f'New Annotation Batch Created - {self.project_name}')

    def _handle_annotation_completed(self, data):
        """Handle annotation completed event with notification"""
        task_id = data.get('task_id')
        annotation_id = data.get('annotation_id')

        logger.info(f'Annotation {annotation_id} completed for task {task_id}')

        # Mark this task as processed
        self.processed_annotations.add(task_id)

        # Send notification if configured
        if self.notification_events.get('annotation_completed', False):
            sample_count = len(self.task_to_samples.get(task_id, []))
            self.send_notification(
                f'Annotation {annotation_id} completed for task {task_id} '
                f'with {sample_count} samples in project {self.project_name}',
                subject=f'Annotation Completed - {self.project_name}')

    def _handle_error(self, data):
        """Handle error event with notification"""
        error_message = data.get('message', 'Unknown error')
        task_id = data.get('task_id', 'Unknown task')

        logger.error(f'Error in annotation task {task_id}: {error_message}')

        # Send notification if configured
        if self.notification_events.get('error_occurred', False):
            self.send_notification(
                f'Error in annotation task {task_id}: {error_message}',
                subject=f'Annotation Error - {self.project_name}',
                notification_type='email'  # Always send errors via email
            )

    @abstractmethod
    def _create_tasks_batch(self, tasks_data: List[Dict],
                            sample_ids: List[Any]) -> List[int]:
        """Create multiple tasks in the annotation platform

        Args:
            tasks_data: List of task data
            sample_ids: List of sample IDs corresponding to each task

        Returns:
            List[int]: List of created task IDs
        """
        pass

    @abstractmethod
    def _format_task(self, samples: List[Dict]) -> Dict:
        """Format samples as an annotation task

        Args:
            samples: List of samples to include in the task

        Returns:
            Dict: Formatted task data
        """
        pass

    @abstractmethod
    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available"""
        pass

    def process_batched(self, samples):
        """Process a batch of samples by creating annotation tasks

        Args:
            samples: Dictionary of samples to process

        Returns:
            Dict: Processed samples
        """
        # Extract sample list from the batch dictionary
        keys = samples.keys()
        first_key = next(iter(keys))
        num_samples = len(samples[first_key])

        # Create a list of sample dictionaries
        sample_list = []
        for i in range(num_samples):
            this_sample = {key: samples[key][i] for key in keys}
            sample_list.append(this_sample)

        # Generate unique IDs for each sample
        sample_ids = [str(uuid.uuid4()) for _ in range(len(sample_list))]

        # Group samples into tasks based on samples_per_task
        tasks_data = []
        task_sample_ids = []

        for i in range(0, len(sample_list), self.samples_per_task):
            batch_samples = sample_list[i:i + self.samples_per_task]
            batch_ids = sample_ids[i:i + self.samples_per_task]

            # Format the samples as a task
            task_data = self._format_task(batch_samples)
            tasks_data.append(task_data)
            task_sample_ids.append(batch_ids)

            # If we've reached max_tasks_per_batch or this is the last group,
            # create the tasks in the annotation platform
            if len(
                    tasks_data
            ) >= self.max_tasks_per_batch or i + self.samples_per_task >= len(
                    sample_list):
                self._create_and_process_batch(tasks_data, task_sample_ids,
                                               sample_list, sample_ids)
                tasks_data = []
                task_sample_ids = []

        # Update the samples with task IDs
        for i, sample_id in enumerate(sample_ids):
            if sample_id in self.sample_to_task_id:
                task_id = self.sample_to_task_id[sample_id]

                # Add task ID to sample metadata
                if Fields.meta not in sample_list[i]:
                    sample_list[i][Fields.meta] = {}
                sample_list[i][Fields.meta]['annotation_task_id'] = task_id

                # If waiting for annotations and they're available, add them
                if (self.wait_for_annotations
                        and task_id in self.processed_annotations):
                    annotation = self._get_task_annotation(task_id)
                    if annotation:
                        sample_list[i][Fields.meta]['annotations'] = annotation

        # Update the original samples dictionary
        for i in range(num_samples):
            for key in keys:
                if key in sample_list[i]:
                    samples[key][i] = sample_list[i][key]

        return samples

    def _create_and_process_batch(self, tasks_data, task_sample_ids,
                                  sample_list, sample_ids):
        """Create a batch of tasks and process the results"""
        if not tasks_data:
            return

        # Generate a batch ID
        batch_id = f'batch_{self.batch_counter}_{int(time.time())}'
        self.batch_counter += 1

        # Flatten the sample IDs for this batch
        all_batch_sample_ids = [id for ids in task_sample_ids for id in ids]

        try:
            # Create the tasks in the annotation platform
            task_ids = self._create_tasks_batch(tasks_data,
                                                all_batch_sample_ids)

            # Map sample IDs to task IDs
            for task_id, sample_id_list in zip(task_ids, task_sample_ids):
                for sample_id in sample_id_list:
                    self.sample_to_task_id[sample_id] = task_id
                self.task_to_samples[task_id] = sample_id_list

                # Trigger task created event
                self.trigger_event(ANNOTATION_EVENTS['TASK_CREATED'], {
                    'task_id': task_id,
                    'sample_ids': sample_id_list
                })

            # Trigger batch created event
            self.trigger_event(
                ANNOTATION_EVENTS['BATCH_CREATED'], {
                    'batch_id': batch_id,
                    'task_count': len(task_ids),
                    'sample_count': len(all_batch_sample_ids)
                })

            # If waiting for annotations, start polling for them
            if self.wait_for_annotations:
                self.start_polling(ANNOTATION_EVENTS['ANNOTATION_COMPLETED'],
                                   self._poll_for_completed_annotations,
                                   interval=self.poll_interval)

                # Wait for all tasks in this batch to be annotated
                try:
                    self._wait_for_batch_annotations(task_ids)
                except TimeoutError as e:
                    # Trigger error event but continue processing
                    self.trigger_event(
                        ANNOTATION_EVENTS['ERROR_OCCURRED'], {
                            'message':
                            f'Timeout waiting for batch annotations: {str(e)}',
                            'batch_id': batch_id
                        })
        except Exception as e:
            # Trigger error event
            self.trigger_event(
                ANNOTATION_EVENTS['ERROR_OCCURRED'], {
                    'message': f'Failed to create batch: {str(e)}',
                    'batch_id': batch_id
                })
            logger.error(f'Error creating batch {batch_id}: {str(e)}')

    def _wait_for_batch_annotations(self, task_ids):
        """Wait for all tasks in a batch to be annotated

        Args:
            task_ids: List of task IDs to wait for

        Raises:
            TimeoutError: If waiting times out
        """
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            # Check if all tasks are annotated
            all_annotated = True
            for task_id in task_ids:
                if task_id not in self.processed_annotations:
                    all_annotated = False
                    break

            if all_annotated:
                return

            # Sleep before checking again
            time.sleep(self.poll_interval)

        # If we get here, we timed out
        raise TimeoutError(
            f'Timed out waiting for annotations of {len(task_ids)} tasks')

    def _poll_for_completed_annotations(self):
        """Poll for completed annotations

        Returns:
            Dict or None: Event data if annotation was done, None otherwise
        """
        # Get all tasks that we're waiting for annotations
        pending_task_ids = [
            task_id for task_id in self.task_to_samples.keys()
            if task_id not in self.processed_annotations
        ]

        if not pending_task_ids:
            return None

        # Check for completed annotations
        for task_id in pending_task_ids:
            try:
                annotation = self._get_task_annotation(task_id)

                # If we have annotations and they weren't processed yet
                if annotation and task_id not in self.processed_annotations:
                    self.processed_annotations.add(task_id)

                    # Return the completed annotation as an event
                    return {
                        'task_id': task_id,
                        'annotation_id': annotation.get('id', 'unknown'),
                        'annotation': annotation,
                        'sample_ids': self.task_to_samples.get(task_id, [])
                    }
            except Exception as e:
                # Trigger error event
                self.trigger_event(ANNOTATION_EVENTS['ERROR_OCCURRED'], {
                    'task_id': task_id,
                    'message': str(e)
                })

        return None

    def __del__(self):
        """Clean up resources when the object is deleted"""
        # Stop all polling threads
        self.stop_all_polling()


class LabelStudioAnnotationMapper(BaseAnnotationMapper):
    """Operation for annotating data using Label Studio"""

    def __init__(self,
                 config_path: Optional[str] = None,
                 api_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 project_name: str = 'Annotation Project',
                 project_id: Optional[int] = None,
                 label_config: Optional[str] = None,
                 wait_for_annotations: bool = False,
                 timeout: int = 3600,
                 poll_interval: int = 60,
                 samples_per_task: int = 1,
                 max_tasks_per_batch: int = 100,
                 notification_config: Optional[Dict] = None,
                 **kwargs):
        """Initialize the Label Studio annotation operation

        Args:
            config_path: Path to the configuration file
            api_url: Base URL for Label Studio API
            api_key: API key for authentication
            project_name: Name of the project to create or use
            project_id: ID of existing project (if None, creates new project)
            label_config: XML configuration for the labeling interface
            wait_for_annotations: Whether to wait for annotations to complete
            timeout: Maximum time to wait for annotations in seconds
            poll_interval: Time between annotation status checks in seconds
            samples_per_task: Number of samples in each annotation task
            max_tasks_per_batch: Maximum number of tasks in a single batch
            notification_config: Configuration for notifications
        """
        super().__init__(config_path=config_path,
                         project_name=project_name,
                         project_id=project_id,
                         wait_for_annotations=wait_for_annotations,
                         timeout=timeout,
                         poll_interval=poll_interval,
                         samples_per_task=samples_per_task,
                         max_tasks_per_batch=max_tasks_per_batch,
                         notification_config=notification_config,
                         **kwargs)

        # Store Label Studio specific configuration
        self.api_url = api_url
        self.api_key = api_key
        self.label_config = label_config

        # Initialize session and project
        self.session = self._create_session()
        if self.project_id is None:
            self.project_id = self.setup_project()

    def _create_session(self) -> requests.Session:
        """Create a session with authentication headers"""
        session = requests.Session()
        session.headers.update({
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        })
        return session

    def setup_project(self) -> int:
        """Create a new project or use existing one"""
        # Create new project
        project_data = {
            'title': self.project_name,
            'description': 'Created by Data Juicer',
            'label_config': self.label_config,
        }

        response = self.session.post(f'{self.api_url}/api/projects',
                                     json=project_data)
        if response.status_code != 201:
            raise Exception(f'Failed to create project: {response.text}')

        project_id = response.json()['id']
        logger.info(f'Created new Label Studio project with ID: {project_id}')

        return project_id

    def _create_tasks_batch(self, tasks_data: List[Dict],
                            sample_ids: List[Any]) -> List[int]:
        """Create multiple tasks in Label Studio

        Args:
            tasks_data: List of task data
            sample_ids: List of sample IDs corresponding to each task

        Returns:
            List[int]: List of created task IDs
        """
        # Label Studio API expects a list of tasks
        response = self.session.post(
            f'{self.api_url}/api/projects/{self.project_id}/import',
            json=tasks_data)

        if response.status_code != 201:
            raise Exception(f'Failed to create tasks: {response.text}')

        result = response.json()
        task_ids = [task['id'] for task in result['tasks']]

        return task_ids

    def _format_task(self, samples: List[Dict]) -> Dict:
        """Format samples as a Label Studio task

        Args:
            samples: List of samples to include in the task

        Returns:
            Dict: Formatted task data
        """
        # For Label Studio, we'll create a task with multiple samples
        task = {'data': {}}

        # If there's only one sample, format it normally
        if len(samples) == 1:
            sample = samples[0]

            # Handle text data
            if self.text_key in sample:
                task['data']['text'] = sample[self.text_key]

            # Handle image data
            if self.image_key in sample and sample[self.image_key]:
                task['data']['image'] = sample[self.image_key][
                    0]  # Use first image

            # Handle audio data
            if self.audio_key in sample and sample[self.audio_key]:
                task['data']['audio'] = sample[self.audio_key][
                    0]  # Use first audio

            # Add any other fields as metadata
            for key, value in sample.items():
                if key not in [
                        self.text_key, self.image_key, self.audio_key,
                        Fields.meta
                ]:
                    # Skip complex objects that can't be serialized
                    if isinstance(value,
                                  (str, int, float, bool)) or value is None:
                        task['data'][f'meta:{key}'] = value
        else:
            # For multiple samples, create a list of items
            task['data']['items'] = []

            for i, sample in enumerate(samples):
                item = {}

                # Handle text data
                if self.text_key in sample:
                    item['text'] = sample[self.text_key]

                # Handle image data
                if self.image_key in sample and sample[self.image_key]:
                    item['image'] = sample[self.image_key][
                        0]  # Use first image

                # Handle audio data
                if self.audio_key in sample and sample[self.audio_key]:
                    item['audio'] = sample[self.audio_key][
                        0]  # Use first audio

                # Add sample index
                item['index'] = i

                # Add any other fields as metadata
                for key, value in sample.items():
                    if key not in [
                            self.text_key, self.image_key, self.audio_key,
                            Fields.meta
                    ]:
                        # Skip complex objects that can't be serialized
                        if isinstance(value,
                                      (str, int, float,
                                       bool)) or value is None:  # noqa: E501
                            item[f'meta:{key}'] = value

                task['data']['items'].append(item)

        return task

    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available

        Args:
            task_id: ID of the task

        Returns:
            Optional[Dict]: Annotation data or None if not yet annotated
        """
        response = self.session.get(f'{self.api_url}/api/tasks/{task_id}')

        if response.status_code != 200:
            raise Exception(f'Failed to get task: {response.text}')

        task = response.json()

        # Check if task has annotations
        if task['annotations'] and len(task['annotations']) > 0:
            return task['annotations'][0]  # Return the first annotation

        return None

    def get_all_annotations(self) -> Dict[int, Dict]:
        """Get all annotations for tasks created by this operation

        Returns:
            Dict[int, Dict]: Dictionary mapping task IDs to annotations
        """
        task_ids = list(self.task_to_samples.keys())

        annotations = {}
        for task_id in task_ids:
            annotation = self._get_task_annotation(task_id)
            if annotation:
                annotations[task_id] = annotation

        return annotations
