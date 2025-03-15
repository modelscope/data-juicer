import unittest
from unittest.mock import patch, MagicMock, call
import json

from data_juicer.ops.mapper.annotation.annotation_mapper import (
    BaseAnnotationOp, 
    LabelStudioAnnotationOp,
    ANNOTATION_EVENTS
)
from data_juicer.utils.constant import Fields


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        self.text = json.dumps(json_data)
    
    def json(self):
        return self._json_data


class MockAnnotationOp(BaseAnnotationOp):
    """Mock implementation of BaseAnnotationOp for testing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.created_tasks = []
        self.mock_annotations = {}
    
    def _create_tasks_batch(self, tasks_data, sample_ids):
        """Mock implementation that returns fake task IDs"""
        task_ids = list(range(1000, 1000 + len(tasks_data)))
        self.created_tasks.extend(zip(task_ids, tasks_data, sample_ids))
        return task_ids
    
    def _format_task(self, samples):
        """Simple mock implementation"""
        return {"samples": [s.get(self.text_key, "") for s in samples]}
    
    def _get_task_annotation(self, task_id):
        """Return mock annotation if available"""
        if task_id in self.mock_annotations:
            return self.mock_annotations[task_id]
        return None
    
    def add_mock_annotation(self, task_id, annotation):
        """Add a mock annotation for testing"""
        self.mock_annotations[task_id] = annotation
        self.processed_annotations.add(task_id)


class TestBaseAnnotationOp(unittest.TestCase):
    
    def setUp(self):
        # Create a mock annotation operator
        self.op = MockAnnotationOp(
            project_name="Test Project",
            samples_per_task=2,
            max_tasks_per_batch=3,
            wait_for_annotations=False
        )
        
        # Sample test data
        self.samples = {
            "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
            "id": [1, 2, 3, 4, 5],
            Fields.meta: [{} for _ in range(5)]
        }
    
    @patch.object(BaseAnnotationOp, 'send_notification')
    @patch.object(BaseAnnotationOp, 'trigger_event')
    def test_process_batched(self, mock_trigger_event, mock_send_notification):
        """Test batch processing of samples"""
        # Process the samples
        result = self.op.process_batched(self.samples)
        
        # Check that the correct number of tasks were created
        # 5 samples with 2 samples per task = 3 tasks (2+2+1)
        self.assertEqual(len(self.op.created_tasks), 3)
        
        # Check that task IDs were added to sample metadata
        for i in range(5):
            self.assertIn('annotation_task_id', result[Fields.meta][i])
        
        # Check that events were triggered
        self.assertEqual(mock_trigger_event.call_count, 4)  # 3 task_created + 1 batch_created
        
        # Verify the batch_created event was triggered with correct data
        batch_created_calls = [
            call for call in mock_trigger_event.call_args_list 
            if call[0][0] == ANNOTATION_EVENTS['BATCH_CREATED']
        ]
        self.assertEqual(len(batch_created_calls), 1)
        batch_event_data = batch_created_calls[0][0][1]
        self.assertEqual(batch_event_data['task_count'], 3)
        self.assertEqual(batch_event_data['sample_count'], 5)
    
    @patch.object(BaseAnnotationOp, 'send_notification')
    def test_handle_task_created(self, mock_send_notification):
        """Test task created event handler with notification"""
        # Enable notifications for task creation
        self.op.notification_events['task_created'] = True
        
        # Trigger the event handler
        self.op._handle_task_created({
            'task_id': 1001,
            'sample_ids': [1, 2]
        })
        
        # Check that notification was sent
        mock_send_notification.assert_called_once()
        self.assertIn("Task 1001 created", mock_send_notification.call_args[0][0])
    
    @patch.object(BaseAnnotationOp, 'send_notification')
    def test_handle_annotation_completed(self, mock_send_notification):
        """Test annotation completed event handler with notification"""
        # Enable notifications for annotation completion
        self.op.notification_events['annotation_completed'] = True
        
        # Set up task-sample mapping
        self.op.task_to_samples[1001] = [1, 2]
        
        # Trigger the event handler
        self.op._handle_annotation_completed({
            'task_id': 1001,
            'annotation_id': 'anno-123'
        })
        
        # Check that notification was sent
        mock_send_notification.assert_called_once()
        self.assertIn("Annotation anno-123 completed", mock_send_notification.call_args[0][0])
        
        # Check that task was marked as processed
        self.assertIn(1001, self.op.processed_annotations)
    
    @patch.object(BaseAnnotationOp, 'send_notification')
    def test_handle_error(self, mock_send_notification):
        """Test error event handler with notification"""
        # Enable notifications for errors
        self.op.notification_events['error_occurred'] = True
        
        # Trigger the event handler
        self.op._handle_error({
            'task_id': 1001,
            'message': 'Test error message'
        })
        
        # Check that notification was sent
        mock_send_notification.assert_called_once()
        self.assertIn("Error in annotation task 1001", mock_send_notification.call_args[0][0])
        self.assertEqual(mock_send_notification.call_args[1]['notification_type'], 'email')
    
    def test_wait_for_annotations(self):
        """Test waiting for annotations"""
        # Set up operator to wait for annotations
        self.op.wait_for_annotations = True
        self.op.timeout = 1  # Short timeout for testing
        self.op.poll_interval = 0.1
        
        # Process samples
        result = self.op.process_batched(self.samples)
        
        # Add mock annotations for the first two tasks
        task_ids = [result[Fields.meta][i]['annotation_task_id'] for i in range(4)]
        self.op.add_mock_annotation(task_ids[0], {"id": "anno-1", "result": "Positive"})
        self.op.add_mock_annotation(task_ids[1], {"id": "anno-2", "result": "Negative"})
        
        # Check that annotations were added to sample metadata
        self.assertIn('annotations', result[Fields.meta][0])
        self.assertIn('annotations', result[Fields.meta][2])
        self.assertEqual(result[Fields.meta][0]['annotations']['id'], "anno-1")
        self.assertEqual(result[Fields.meta][2]['annotations']['id'], "anno-2")
        
        # The last task doesn't have annotations, so it shouldn't have the field
        self.assertNotIn('annotations', result[Fields.meta][4])


class TestLabelStudioAnnotationOp(unittest.TestCase):
    
    def setUp(self):
        # Create a patcher for requests.Session
        self.session_patcher = patch('requests.Session')
        self.mock_session = self.session_patcher.start()
        
        # Mock the session instance
        self.mock_session_instance = MagicMock()
        self.mock_session.return_value = self.mock_session_instance
        
        # Create the operator
        self.op = LabelStudioAnnotationOp(
            api_url="http://localhost:8080",
            api_key="test_api_key",
            project_name="Test Project",
            project_id=None,  # Will create a new project
            label_config="<View><Text name='text' value='$text'/></View>",
            samples_per_task=2,
            max_tasks_per_batch=3
        )
    
    def tearDown(self):
        self.session_patcher.stop()
    
    def test_create_session(self):
        """Test session creation with auth headers"""
        # Check that headers were set correctly
        headers = self.mock_session_instance.headers.update.call_args[0][0]
        self.assertEqual(headers["Authorization"], "Token test_api_key")
        self.assertEqual(headers["Content-Type"], "application/json")
    
    def test_setup_project(self):
        """Test project creation"""
        # Mock the response for project creation
        self.mock_session_instance.post.return_value = MockResponse(
            201, {"id": 42, "title": "Test Project"}
        )
        
        # Call setup_project
        project_id = self.op.setup_project()
        
        # Check that the correct API was called
        self.mock_session_instance.post.assert_called_with(
            "http://localhost:8080/api/projects", 
            json={
                "title": "Test Project",
                "description": "Created by Data Juicer",
                "label_config": "<View><Text name='text' value='$text'/></View>"
            }
        )
        
        # Check that project ID was returned
        self.assertEqual(project_id, 42)
    
    def test_create_tasks_batch(self):
        """Test batch creation of tasks"""
        # Mock the response for task creation
        self.mock_session_instance.post.return_value = MockResponse(
            201, {
                "tasks": [
                    {"id": 101, "data": {"text": "Sample 1"}},
                    {"id": 102, "data": {"text": "Sample 2"}}
                ]
            }
        )
        
        # Create tasks
        tasks_data = [
            {"data": {"text": "Sample 1"}},
            {"data": {"text": "Sample 2"}}
        ]
        sample_ids = [1, 2]
        
        task_ids = self.op._create_tasks_batch(tasks_data, sample_ids)
        
        # Check that the correct API was called
        self.mock_session_instance.post.assert_called_with(
            "http://localhost:8080/api/projects/42/import",
            json=tasks_data
        )
        
        # Check that task IDs were returned
        self.assertEqual(task_ids, [101, 102])
    
    def test_format_task_single_sample(self):
        """Test formatting a single sample as a task"""
        # Create a sample
        sample = {
            "text": "This is a test sample",
            "label": "test",
            Fields.meta: {"source": "test"}
        }
        
        # Format the task
        task = self.op._format_task([sample])
        
        # Check the task format
        self.assertEqual(task["data"]["text"], "This is a test sample")
        self.assertEqual(task["data"]["meta:label"], "test")
    
    def test_format_task_multiple_samples(self):
        """Test formatting multiple samples as a task"""
        # Create samples
        samples = [
            {"text": "Sample 1", "label": "test1"},
            {"text": "Sample 2", "label": "test2"}
        ]
        
        # Format the task
        task = self.op._format_task(samples)
        
        # Check the task format
        self.assertEqual(len(task["data"]["items"]), 2)
        self.assertEqual(task["data"]["items"][0]["text"], "Sample 1")
        self.assertEqual(task["data"]["items"][0]["meta:label"], "test1")
        self.assertEqual(task["data"]["items"][1]["text"], "Sample 2")
        self.assertEqual(task["data"]["items"][1]["meta:label"], "test2")
    
    def test_get_task_annotation(self):
        """Test getting annotation for a task"""
        # Mock the response for getting a task
        self.mock_session_instance.get.return_value = MockResponse(
            200, {
                "id": 101,
                "data": {"text": "Sample 1"},
                "annotations": [
                    {
                        "id": 201,
                        "result": [{"value": {"choices": ["Positive"]}}]
                    }
                ]
            }
        )
        
        # Get annotation
        annotation = self.op._get_task_annotation(101)
        
        # Check that the correct API was called
        self.mock_session_instance.get.assert_called_with(
            "http://localhost:8080/api/tasks/101"
        )
        
        # Check that annotation was returned
        self.assertEqual(annotation["id"], 201)
        self.assertEqual(annotation["result"][0]["value"]["choices"], ["Positive"])
    
    def test_get_task_annotation_not_annotated(self):
        """Test getting annotation for a task that hasn't been annotated yet"""
        # Mock the response for getting a task without annotations
        self.mock_session_instance.get.return_value = MockResponse(
            200, {
                "id": 101,
                "data": {"text": "Sample 1"},
                "annotations": []
            }
        )
        
        # Get annotation
        annotation = self.op._get_task_annotation(101)
        
        # Check that None was returned
        self.assertIsNone(annotation)
    
    @patch.object(LabelStudioAnnotationOp, 'trigger_event')
    def test_process_batched_integration(self, mock_trigger_event):
        """Test the full process_batched method with mocked API calls"""
        # Mock project creation
        self.mock_session_instance.post.side_effect = [
            # Project creation response
            MockResponse(201, {"id": 42, "title": "Test Project"}),
            # First batch of tasks
            MockResponse(201, {
                "tasks": [
                    {"id": 101, "data": {"text": "Sample 1, Sample 2"}},
                    {"id": 102, "data": {"text": "Sample 3, Sample 4"}},
                    {"id": 103, "data": {"text": "Sample 5"}}
                ]
            })
        ]
        
        # Sample test data
        samples = {
            "text": ["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
            "id": [1, 2, 3, 4, 5],
            Fields.meta: [{} for _ in range(5)]
        }
        
        # Process the samples
        result = self.op.process_batched(samples)
        
        # Check that task IDs were added to sample metadata
        for i in range(5):
            self.assertIn('annotation_task_id', result[Fields.meta][i])
        
        # Check that the correct events were triggered
        self.assertEqual(mock_trigger_event.call_count, 4)  # 3 task_created + 1 batch_created
        
        # Verify the batch_created event was triggered with correct data
        batch_created_calls = [
            call for call in mock_trigger_event.call_args_list 
            if call[0][0] == ANNOTATION_EVENTS['BATCH_CREATED']
        ]
        self.assertEqual(len(batch_created_calls), 1)
        batch_event_data = batch_created_calls[0][0][1]
        self.assertEqual(batch_event_data['task_count'], 3)
        self.assertEqual(batch_event_data['sample_count'], 5)


if __name__ == '__main__':
    unittest.main() 