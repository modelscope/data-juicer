import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Any
import time

from data_juicer.ops.mapper.annotation.annotation_mapper import (
    BaseAnnotationMapper, ANNOTATION_EVENTS
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MockAnnotationMapper(BaseAnnotationMapper):
    """Mock implementation of BaseAnnotationMapper for testing"""
    _name = "MockAnnotationMapper"  # Default name for test classes
    _batched_op = True

    def __init__(self, **kwargs):
        # Set the name attribute before calling parent initializers
        super().__init__(**kwargs)
        self.mock_tasks = {}
        self.mock_annotations = {}
        self.created_task_ids = []

    def _create_tasks_batch(self, tasks_data: List[Dict],
                           sample_ids: List[Any]) -> List[int]:
        """Mock implementation that returns fake task IDs"""
        task_ids = []
        for i, (task_data, sample_id_list) in enumerate(zip(tasks_data, sample_ids)):
            task_id = i + 1000  # Start with task ID 1000
            self.mock_tasks[task_id] = task_data
            task_ids.append(task_id)
            self.created_task_ids.append(task_id)
        return task_ids

    def _format_task(self, samples: List[Dict]) -> Dict:
        """Mock implementation that simply returns the samples as a task"""
        return {"samples": samples}

    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Mock implementation to return annotations if they exist"""
        return self.mock_annotations.get(task_id)

    def _process_annotation_result(self, annotation: Dict, sample: Dict) -> Dict:
        """Mock implementation that adds annotation to the sample"""
        sample_copy = sample.copy()
        sample_copy["annotation_result"] = annotation.get("result", {})
        return sample_copy

    def _check_annotation_status(self, task_ids):
        """Mock implementation for checking annotation status"""
        has_changes = False
        completed_tasks = {}

        for task_id in task_ids:
            if task_id in self.mock_annotations and task_id not in self.processed_annotations:
                has_changes = True
                completed_tasks[task_id] = self.mock_annotations[task_id]

        return has_changes, completed_tasks

    def add_mock_annotation(self, task_id, annotation_data):
        """Helper method to add mock annotations for testing"""
        self.mock_annotations[task_id] = {
            "id": f"annotation_{task_id}",
            "result": annotation_data
        }


class AnnotationMapperTest(DataJuicerTestCaseBase):
    """Test cases for the BaseAnnotationMapper"""

    def setUp(self):
        # Create samples for testing
        self.samples = [
            {"text": "Sample 1 text", "id": "sample1"},
            {"text": "Sample 2 text", "id": "sample2"},
            {"text": "Sample 3 text", "id": "sample3"},
            {"text": "Sample 4 text", "id": "sample4"},
            {"text": "Sample 5 text", "id": "sample5"},
        ]
        
        # Create a dictionary version of samples (column-oriented)
        self.samples_dict = {
            "text": [s["text"] for s in self.samples],
            "id": [s["id"] for s in self.samples]
        }

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        mapper = MockAnnotationMapper()
        self.assertFalse(mapper.wait_for_annotations)
        self.assertEqual(mapper.timeout, 3600)
        self.assertEqual(mapper.poll_interval, 60)
        self.assertEqual(mapper.samples_per_task, 1)
        self.assertEqual(mapper.max_tasks_per_batch, 100)
        self.assertIsNone(mapper.project_id)
        self.assertTrue(mapper.project_name.startswith('DataJuicer_Annotation_'))
        self.assertEqual(mapper.notification_events, {
            'task_created': False,
            'batch_created': True,
            'annotation_completed': False,
            'batch_annotation_completed': True,
            'error_occurred': True
        })

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters"""
        custom_notification_events = {
            'task_created': True,
            'batch_created': False,
            'annotation_completed': True,
            'error_occurred': False
        }
        
        mapper = MockAnnotationMapper(
            project_name_prefix="TestProject",
            wait_for_annotations=True,
            timeout=1800,
            poll_interval=30,
            samples_per_task=2,
            max_tasks_per_batch=50,
            project_id=123,
            notification_events=custom_notification_events
        )
        
        self.assertTrue(mapper.wait_for_annotations)
        self.assertEqual(mapper.timeout, 1800)
        self.assertEqual(mapper.poll_interval, 30)
        self.assertEqual(mapper.samples_per_task, 2)
        self.assertEqual(mapper.max_tasks_per_batch, 50)
        self.assertEqual(mapper.project_id, 123)
        self.assertTrue(mapper.project_name.startswith('TestProject_'))
        self.assertEqual(mapper.notification_events, custom_notification_events)

    def test_event_handlers_registration(self):
        """Test that event handlers are properly registered"""
        mapper = MockAnnotationMapper()
        
        # Check that all event handlers are registered
        self.assertIn(ANNOTATION_EVENTS['TASK_CREATED'], mapper.event_handlers)
        self.assertIn(ANNOTATION_EVENTS['BATCH_CREATED'], mapper.event_handlers)
        self.assertIn(ANNOTATION_EVENTS['ANNOTATION_COMPLETED'], mapper.event_handlers)
        self.assertIn(ANNOTATION_EVENTS['ERROR_OCCURRED'], mapper.event_handlers)
        
        # Each event should have exactly one handler
        self.assertEqual(len(mapper.event_handlers[ANNOTATION_EVENTS['TASK_CREATED']]), 1)
        self.assertEqual(len(mapper.event_handlers[ANNOTATION_EVENTS['BATCH_CREATED']]), 1)
        self.assertEqual(len(mapper.event_handlers[ANNOTATION_EVENTS['ANNOTATION_COMPLETED']]), 1)
        self.assertEqual(len(mapper.event_handlers[ANNOTATION_EVENTS['ERROR_OCCURRED']]), 1)

    @patch('data_juicer.ops.mapper.annotation.annotation_mapper.BaseAnnotationMapper.send_notification')
    def test_task_created_handler(self, mock_send_notification):
        """Test task created event handler"""
        mapper = MockAnnotationMapper()
        
        # Test without notification
        task_data = {
            'task_id': 123,
            'sample_ids': ['sample1', 'sample2']
        }
        mapper._handle_task_created(task_data)
        mock_send_notification.assert_not_called()
        
        # Test with notification
        mapper.notification_events['task_created'] = True
        mapper._handle_task_created(task_data)
        mock_send_notification.assert_called_once()

    @patch('data_juicer.ops.mapper.annotation.annotation_mapper.BaseAnnotationMapper.send_notification')
    def test_batch_created_handler(self, mock_send_notification):
        """Test batch created event handler"""
        mapper = MockAnnotationMapper()
        
        # Test with notification (enabled by default)
        batch_data = {
            'batch_id': 'batch_123',
            'task_count': 10,
            'sample_count': 20
        }
        mapper._handle_batch_created(batch_data)
        mock_send_notification.assert_called_once()
        
        # Test without notification
        mock_send_notification.reset_mock()
        mapper.notification_events['batch_created'] = False
        mapper._handle_batch_created(batch_data)
        mock_send_notification.assert_not_called()

    @patch('data_juicer.ops.mapper.annotation.annotation_mapper.BaseAnnotationMapper.send_notification')
    def test_annotation_completed_handler(self, mock_send_notification):
        """Test annotation completed event handler"""
        mapper = MockAnnotationMapper()
        
        # Test without notification (disabled by default)
        task_id = 123
        sample_ids = ['sample1', 'sample2']
        mapper.task_to_samples[task_id] = sample_ids
        
        annotation_data = {
            'task_id': task_id,
            'annotation_id': 'annotation_123'
        }
        mapper._handle_annotation_completed(annotation_data)
        
        # Verify notification was not sent (disabled by default)
        mock_send_notification.assert_not_called()
        
        # Test with notification enabled
        mock_send_notification.reset_mock()
        mapper.notification_events['annotation_completed'] = True
        mapper._handle_annotation_completed(annotation_data)
        
        # Verify notification was sent
        mock_send_notification.assert_called_once()
        
        # Verify task was marked as processed
        self.assertIn(task_id, mapper.processed_annotations)

    @patch('data_juicer.ops.mapper.annotation.annotation_mapper.BaseAnnotationMapper.send_notification')
    def test_error_handler(self, mock_send_notification):
        """Test error event handler"""
        mapper = MockAnnotationMapper()
        
        # Test with notification (enabled by default)
        error_data = {
            'task_id': 123,
            'message': 'Test error message'
        }
        mapper._handle_error(error_data)
        
        # Verify notification was sent, and it was set to email
        mock_send_notification.assert_called_once_with(
            'Error in annotation task 123: Test error message',
            subject='Annotation Error - ' + mapper.project_name,
            notification_type='email'
        )
        
        # Test without notification
        mock_send_notification.reset_mock()
        mapper.notification_events['error_occurred'] = False
        mapper._handle_error(error_data)
        mock_send_notification.assert_not_called()

    def test_process_batched_without_waiting(self):
        """Test processing a batch of samples without waiting for annotations"""
        mapper = MockAnnotationMapper(wait_for_annotations=False)
        
        # Process the samples
        result = mapper.process_batched(self.samples_dict)
        
        # Verify results
        self.assertEqual(len(result["text"]), len(self.samples_dict["text"]))
        self.assertEqual(len(result["id"]), len(self.samples_dict["id"]))
        
        # Verify tasks were created
        self.assertEqual(len(mapper.created_task_ids), 5)  # One task per sample with default settings
        
        # Verify sample to task mappings were created
        self.assertEqual(len(mapper.sample_to_task_id), 5)
        self.assertEqual(len(mapper.task_to_samples), 5)

    def test_process_batched_with_waiting(self):
        """Test processing a batch of samples and waiting for annotations"""
        mapper = MockAnnotationMapper(wait_for_annotations=True)
        
        # Add mock annotations for all tasks that will be created
        for i in range(5):
            task_id = 1000 + i  # Matches the mock implementation's ID generation
            mapper.add_mock_annotation(task_id, {"label": f"Annotation for task {task_id}"})
        
        # Process the samples
        result = mapper.process_batched(self.samples_dict)
        
        # Verify results
        self.assertEqual(len(result["text"]), len(self.samples_dict["text"]))
        self.assertEqual(len(result["id"]), len(self.samples_dict["id"]))
        
        # Verify annotation results were added to samples
        for i in range(5):
            self.assertIn("annotation_result", result)
            self.assertEqual(result["annotation_result"][i]["label"], f"Annotation for task {1000 + i}")

    def test_process_batched_with_custom_samples_per_task(self):
        """Test processing with multiple samples per task"""
        mapper = MockAnnotationMapper(samples_per_task=2)
        
        # Process the samples
        result = mapper.process_batched(self.samples_dict)
        
        # Verify results
        self.assertEqual(len(result["text"]), len(self.samples_dict["text"]))
        self.assertEqual(len(result["id"]), len(self.samples_dict["id"]))
        
        # Verify tasks were created (5 samples with 2 per task = 3 tasks)
        self.assertEqual(len(mapper.created_task_ids), 3)
        
        # Check sample to task mappings
        # First task should have 2 samples
        first_task_id = mapper.created_task_ids[0]
        self.assertEqual(len(mapper.task_to_samples[first_task_id]), 2)
        
        # Last task should have 1 sample (5th sample)
        last_task_id = mapper.created_task_ids[2]
        self.assertEqual(len(mapper.task_to_samples[last_task_id]), 1)

    def test_wait_for_batch_annotations_timeout(self):
        """Test waiting for annotations with a timeout"""
        # Create a mapper with a very short timeout
        mapper = MockAnnotationMapper(wait_for_annotations=True, timeout=0.1, poll_interval=0.01)
        
        # Create a task but don't add annotations
        task_ids = [1001, 1002, 1003]
        
        # Now let's extract and patch the actual timeout checking logic
        start_time = time.time()
        
        # Replace the _check_annotation_status method to simulate no annotations being completed
        original_check = mapper._check_annotation_status
        
        def mock_check_that_never_completes(*args, **kwargs):
            # Always return no changes and no completed tasks
            return False, {}
            
        mapper._check_annotation_status = mock_check_that_never_completes
        
        try:
            # The method should return when it times out, with empty completed_tasks
            completed_tasks = mapper._wait_for_batch_annotations(task_ids)
            
            # Verify timeout behavior
            self.assertEqual(len(completed_tasks), 0)  # No tasks should be completed
            # Verify enough time has passed (at least close to the timeout)
            elapsed = time.time() - start_time
            self.assertGreaterEqual(elapsed, 0.1 * 0.9)  # Allow a small margin of error
            
        finally:
            # Restore the original method
            mapper._check_annotation_status = original_check

    def test_wait_for_batch_annotations_success(self):
        """Test successful waiting for annotations"""
        mapper = MockAnnotationMapper(wait_for_annotations=True, timeout=1, poll_interval=0.01)
        
        # Create tasks and add annotations
        task_ids = [1001, 1002, 1003]
        for task_id in task_ids:
            mapper.add_mock_annotation(task_id, {"label": f"Annotation for task {task_id}"})
        
        # Wait for annotations (should succeed)
        completed_tasks = mapper._wait_for_batch_annotations(task_ids)
        
        # Verify all tasks were completed
        self.assertEqual(len(completed_tasks), 3)
        for task_id in task_ids:
            self.assertIn(task_id, completed_tasks)
            self.assertEqual(completed_tasks[task_id]["result"]["label"], f"Annotation for task {task_id}")

    def test_process_uses_existing_ids(self):
        """Test that the mapper uses existing IDs in samples instead of generating new ones"""
        # First pass: process without waiting for annotations
        mapper = MockAnnotationMapper(wait_for_annotations=False)
        
        # Create samples with predefined IDs
        samples_with_ids = {
            "text": ["Sample text 1", "Sample text 2", "Sample text 3"],
            "id": ["predefined_id_1", "predefined_id_2", "predefined_id_3"]
        }
        
        # Process the samples
        result = mapper.process_batched(samples_with_ids)
        
        # Verify that the predefined IDs were used in the mapping
        for i, sample_id in enumerate(samples_with_ids["id"]):
            # Check if each predefined ID is in the sample-to-task mapping
            self.assertIn(sample_id, mapper.sample_to_task_id)
            
            # Get the task ID for this sample
            task_id = mapper.sample_to_task_id[sample_id]
            
            # Verify that the sample ID is in the task's sample list
            self.assertIn(sample_id, mapper.task_to_samples[task_id])
            
        # Add mock annotations using the task IDs that were created
        for task_id in mapper.created_task_ids:
            mapper.add_mock_annotation(task_id, {"label": f"Annotation for task {task_id}"})
            
        # Second pass: process with waiting for annotations
        mapper.wait_for_annotations = True
        result = mapper.process_batched(samples_with_ids)
        
        # Verify results include the annotations
        self.assertIn("annotation_result", result)
        for i in range(len(samples_with_ids["id"])):
            task_id = mapper.sample_to_task_id[samples_with_ids["id"][i]]
            self.assertEqual(result["annotation_result"][i]["label"], f"Annotation for task {task_id}")


class MockLabelStudioAnnotationMapper(BaseAnnotationMapper):
    """Mock implementation of LabelStudioAnnotationMapper for testing"""
    _name = "MockLabelStudioAnnotationMapper" 
    _batched_op = True

    def __init__(self, **kwargs):
                # Skip LabelStudioAnnotationMapper initialization to avoid connecting to Label Studio
        # And initialize directly from BaseAnnotationMapper
        BaseAnnotationMapper.__init__(self, **kwargs)
        
        self.mock_tasks = {}
        self.mock_annotations = {}
        self.created_task_ids = []
        
        # Mock Label Studio client and project
        self.client = MagicMock()
        self.project = MagicMock()
        self.project.id = 999
        
        # Make sure samples_per_task is 1 (Label Studio requirement)
        if self.samples_per_task != 1:
            self.samples_per_task = 1
    
    def _create_tasks_batch(self, tasks_data, sample_ids):
        """Mock implementation that returns fake task IDs"""
        task_ids = []
        for i, task_data in enumerate(tasks_data):
            task_id = i + 2000  # Start with task ID 2000
            self.mock_tasks[task_id] = task_data
            task_ids.append(task_id)
            self.created_task_ids.append(task_id)
        return task_ids
    
    def _format_task(self, samples):
        """Mock implementation for Label Studio"""
        # Label Studio format typically has 'data' field
        return {"data": samples[0] if samples else {}}
    
    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Mock implementation to return annotations if they exist"""
        return self.mock_annotations.get(task_id)
    
    def _check_annotation_status(self, task_ids):
        """Mock implementation for checking annotation status"""
        has_changes = False
        completed_tasks = {}

        for task_id in task_ids:
            if task_id in self.mock_annotations and task_id not in self.processed_annotations:
                has_changes = True
                completed_tasks[task_id] = self.mock_annotations[task_id]

        return has_changes, completed_tasks
    
    def _process_annotation_result(self, annotation, sample):
        """Mock implementation for Label Studio"""
        sample_copy = sample.copy()
        sample_copy["label_studio_result"] = annotation.get("result", [])
        return sample_copy
    
    def add_mock_annotation(self, task_id, annotation_data):
        """Helper method to add mock annotations for testing"""
        self.mock_annotations[task_id] = {
            "id": f"annotation_{task_id}",
            "result": annotation_data
        }


class LabelStudioAnnotationMapperTest(DataJuicerTestCaseBase):
    """Test cases for LabelStudioAnnotationMapper"""
    
    def setUp(self):
        # Create samples for testing
        self.samples = [
            {"text": "Label Studio Sample 1", "id": "ls_sample1"},
            {"text": "Label Studio Sample 2", "id": "ls_sample2"},
        ]
        
        # Create a dictionary version of samples (column-oriented)
        self.samples_dict = {
            "text": [s["text"] for s in self.samples],
            "id": [s["id"] for s in self.samples]
        }
    
    def test_samples_per_task_enforcement(self):
        """Test that samples_per_task is always 1 for Label Studio"""
        # Try to create with samples_per_task=2
        mapper = MockLabelStudioAnnotationMapper(samples_per_task=2)
        
        # It should be reset to 1
        self.assertEqual(mapper.samples_per_task, 1)
    
    def test_process_batched(self):
        """Test processing a batch of samples with Label Studio mapper"""
        mapper = MockLabelStudioAnnotationMapper(wait_for_annotations=True)
        
        # Add mock annotations for all tasks that will be created
        for i in range(len(self.samples)):
            task_id = 2000 + i  # Matches the mock implementation's ID generation
            mapper.add_mock_annotation(task_id, [{"value": {"labels": ["Positive"]}}])
        
        # Process the samples
        result = mapper.process_batched(self.samples_dict)
        
        # Verify results
        self.assertEqual(len(result["text"]), len(self.samples_dict["text"]))
        self.assertEqual(len(result["id"]), len(self.samples_dict["id"]))
        
        # Verify annotation results were added to samples
        for i in range(len(self.samples)):
            self.assertIn("label_studio_result", result)
            self.assertEqual(result["label_studio_result"][i][0]["value"]["labels"][0], "Positive")


if __name__ == '__main__':
    unittest.main()
