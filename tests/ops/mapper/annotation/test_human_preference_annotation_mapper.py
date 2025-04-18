import unittest
from unittest.mock import MagicMock
from typing import Dict, List, Optional

from data_juicer.ops.mapper.annotation.annotation_mapper import (
    BaseAnnotationMapper
)
from data_juicer.ops.mapper.annotation.human_preference_annotation_mapper import (
    HumanPreferenceAnnotationMapper
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class MockHumanPreferenceAnnotationMapper(HumanPreferenceAnnotationMapper):
    """Mock implementation of HumanPreferenceAnnotationMapper for testing"""
    _name = "MockHumanPreferenceAnnotationMapper"
    _batched_op = True

    def __init__(self, **kwargs):
        # Skip parent initialization to avoid connecting to Label Studio
        BaseAnnotationMapper.__init__(self, **kwargs)

        # Store HumanPreferenceAnnotationMapper specific parameters
        self.answer1_key = kwargs.get('answer1_key', 'answer1')
        self.answer2_key = kwargs.get('answer2_key', 'answer2')
        self.prompt_key = kwargs.get('prompt_key', 'prompt')
        self.chosen_key = kwargs.get('chosen_key', 'chosen')
        self.rejected_key = kwargs.get('rejected_key', 'rejected')

        # Ensure text_key is set to prompt_key if not explicitly provided
        if 'text_key' not in kwargs:
            self.text_key = self.prompt_key

        # Use default label config
        self.label_config = HumanPreferenceAnnotationMapper.DEFAULT_LABEL_CONFIG

        # Set up other necessary attributes
        self.mock_tasks = {}
        self.mock_annotations = {}
        self.created_task_ids = []

        # Mock Label Studio client and project
        self.client = MagicMock()
        self.project = MagicMock()
        self.project.id = 999

        # Make sure samples_per_task is 1 (Label Studio requirement)
        self.samples_per_task = 1

    def _create_tasks_batch(self, tasks_data, sample_ids):
        """Mock implementation that returns fake task IDs"""
        task_ids = []
        for i, task_data in enumerate(tasks_data):
            task_id = i + 3000  # Start with task ID 3000
            self.mock_tasks[task_id] = task_data
            task_ids.append(task_id)
            self.created_task_ids.append(task_id)
        return task_ids

    def _check_annotation_status(self, task_ids):
        """Mock implementation for checking annotation status"""
        has_changes = False
        completed_tasks = {}

        for task_id in task_ids:
            if task_id in self.mock_annotations and task_id not in self.processed_annotations:
                has_changes = True
                completed_tasks[task_id] = self.mock_annotations[task_id]

        return has_changes, completed_tasks

    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available with preference processing"""
        annotation = self.mock_annotations.get(task_id)

        # Process the annotation if available to extract preference
        if annotation and 'chosen' in annotation:
            # Extract the preference information
            for item in annotation['chosen']:
                if item.get('type') == 'pairwise':
                    # Get the selected option (from_id or to_id)
                    selected = item.get('value', {}).get('selected')
                    if selected:
                        # Add the preference to the annotation
                        annotation['preference'] = selected

        return annotation

    def add_mock_annotation(self, task_id, annotation_data):
        """Helper method to add mock annotations for testing"""
        self.mock_annotations[task_id] = {
            "id": f"annotation_{task_id}",
            "result": annotation_data
        }

    def _format_task(self, samples: List[Dict]) -> Dict:
        """Format samples as a Label Studio task for human preference.

        Args:
            samples: List of samples to include in the task

        Returns:
            Dict: Formatted task data
        """
        # For human preference, we need a special format
        if len(samples) != 1:
            # Only use first sample if multiple provided
            samples = [samples[0]]

        sample = samples[0]
        task = {'data': {}}

        # Add the prompt/question
        if self.prompt_key in sample:
            task['data']['prompt'] = sample[self.prompt_key]
        else:
            task['data']['prompt'] = 'No prompt provided'

        # Add the answer options
        if self.answer1_key in sample:
            task['data']['answer1'] = sample[self.answer1_key]
        else:
            task['data']['answer1'] = 'No answer 1 provided'

        if self.answer2_key in sample:
            task['data']['answer2'] = sample[self.answer2_key]
        else:
            task['data']['answer2'] = 'No answer 2 provided'

        # Add any other metadata as string values
        for key, value in sample.items():
            if key not in [self.prompt_key, self.answer1_key,
                           self.answer2_key]:
                task['data'][f'meta:{key}'] = str(
                    value) if value is not None else ''

        return task

    def _process_annotation_result(self, annotation: Dict,
                                   sample: Dict) -> Dict:
        """Process human preference annotation result and update the sample

        Args:
            annotation: The annotation result from the annotation platform
            sample: The original sample that was annotated

        Returns:
            Dict: The updated sample with preference results
        """
        # Make a copy of the sample to avoid modifying the original
        sample_copy = sample.copy()

        # Extract the preference information
        all_keys = f'{self.answer1_key}{self.answer2_key}'
        preference = None
        for item in annotation['result']:
            if item.get('type') == 'pairwise':
                # Get the selected option
                selected = item.get('value', {}).get('selected')
                if selected:
                    # Map 'left'/'right' to 'answer1'/'answer2'
                    if selected == 'left':
                        preference = self.answer1_key
                    elif selected == 'right':
                        preference = self.answer2_key
                    else:
                        # In case it's already 'answer1'/'answer2'
                        preference = selected
                    break

        # Store the preference result directly in the sample
        chosen = preference if preference else 'Unanswered'
        rejected = all_keys.replace(preference, '') if preference else 'Unanswered'
        sample_copy[self.chosen_key] = sample_copy[chosen]
        sample_copy[self.rejected_key] = sample_copy[rejected]

        return sample_copy


class HumanPreferenceAnnotationMapperTest(DataJuicerTestCaseBase):
    """Test cases for HumanPreferenceAnnotationMapper"""

    def setUp(self):
        # Create samples for testing human preference
        self.samples = [
            {
                "prompt": "Which response is more helpful?",
                "answer1": "The capital of France is Paris.",
                "answer2": "Paris is the capital and largest city of France, located on the Seine River.",
                "id": "pref_sample1"
            },
            {
                "prompt": "Which explanation is clearer?",
                "answer1": "To create a list in Python, use square brackets.",
                "answer2": "In Python, you can create a list using square brackets []. Lists can contain items of different types.",
                "id": "pref_sample2"
            },
        ]

        # Create a dictionary version of samples (column-oriented)
        self.samples_dict = {
            "prompt": [s["prompt"] for s in self.samples],
            "answer1": [s["answer1"] for s in self.samples],
            "answer2": [s["answer2"] for s in self.samples],
            "id": [s["id"] for s in self.samples]
        }

    def test_init_parameters(self):
        """Test initialization with custom parameters"""
        mapper = MockHumanPreferenceAnnotationMapper(
            answer1_key="response_a",
            answer2_key="response_b",
            prompt_key="question",
            chosen_key="chosen",
            rejected_key="rejected",
        )

        self.assertEqual(mapper.answer1_key, "response_a")
        self.assertEqual(mapper.answer2_key, "response_b")
        self.assertEqual(mapper.prompt_key, "question")
        self.assertEqual(mapper.chosen_key, "chosen")
        self.assertEqual(mapper.rejected_key, "rejected")
        self.assertEqual(mapper.text_key,
                         "question")  # text_key should match prompt_key

    def test_format_task(self):
        """Test task formatting for human preference"""
        mapper = MockHumanPreferenceAnnotationMapper()

        # Format a task from the first sample
        formatted_task = mapper._format_task([self.samples[0]])

        # Verify the formatting
        self.assertIn('data', formatted_task)
        self.assertEqual(formatted_task['data']['prompt'],
                         self.samples[0]['prompt'])
        self.assertEqual(formatted_task['data']['answer1'],
                         self.samples[0]['answer1'])
        self.assertEqual(formatted_task['data']['answer2'],
                         self.samples[0]['answer2'])
        self.assertEqual(formatted_task['data']['meta:id'],
                         self.samples[0]['id'])

    def test_process_annotation_result_left_preference(self):
        """Test processing annotation result when left option is preferred"""
        mapper = MockHumanPreferenceAnnotationMapper()

        # Create a sample
        sample = self.samples[0].copy()

        # Create an annotation with preference for the left option (answer1)
        annotation = {
            "id": "annotation_1",
            "result": [
                {
                    "type": "pairwise",
                    "value": {
                        "selected": "left"
                    }
                }
            ]
        }

        # Process the annotation
        processed_sample = mapper._process_annotation_result(annotation,
                                                             sample)

        # Verify the result
        self.assertEqual(processed_sample['chosen'], processed_sample['answer1'])

    def test_process_annotation_result_right_preference(self):
        """Test processing annotation result when right option is preferred"""
        mapper = MockHumanPreferenceAnnotationMapper()

        # Create a sample
        sample = self.samples[0].copy()

        # Create an annotation with preference for the right option (answer2)
        annotation = {
            "id": "annotation_1",
            "result": [
                {
                    "type": "pairwise",
                    "value": {
                        "selected": "right"
                    }
                }
            ]
        }

        # Process the annotation
        processed_sample = mapper._process_annotation_result(annotation,
                                                             sample)

        # Verify the result
        self.assertEqual(processed_sample['chosen'], processed_sample['answer2'])

    def test_process_batched(self):
        """Test processing a batch of samples with HumanPreferenceAnnotationMapper"""
        mapper = MockHumanPreferenceAnnotationMapper(wait_for_annotations=True)

        # Add mock annotations for all tasks that will be created
        for i in range(len(self.samples)):
            task_id = 3000 + i  # Matches the mock implementation's ID generation

            # Alternate between answer1 and answer2 for testing
            selected = "left" if i % 2 == 0 else "right"

            mapper.add_mock_annotation(task_id, [
                {
                    "type": "pairwise",
                    "value": {
                        "selected": selected
                    }
                }
            ])

        # Process the samples
        result = mapper.process_batched(self.samples_dict)

        # Verify results
        self.assertEqual(len(result["prompt"]),
                         len(self.samples_dict["prompt"]))
        self.assertEqual(len(result["id"]), len(self.samples_dict["id"]))

        # Verify that results are properly added
        self.assertIn("chosen", result)

        # First sample should prefer answer1 (left choice)
        self.assertEqual(result["chosen"][0], result["answer1"][0])

        # Second sample should prefer answer2 (right choice)
        self.assertEqual(result["chosen"][1],result["answer2"][1])

    def test_custom_keys(self):
        """Test using custom keys for answers and prompt"""
        # Create a sample with custom keys
        custom_sample = {
            "question": "Which explanation is better?",
            "response_a": "Explanation A",
            "response_b": "Explanation B",
            "id": "custom_sample"
        }

        # Create a dictionary version of the sample
        custom_dict = {
            "question": [custom_sample["question"]],
            "response_a": [custom_sample["response_a"]],
            "response_b": [custom_sample["response_b"]],
            "id": [custom_sample["id"]]
        }

        # Create mapper with custom keys
        mapper = MockHumanPreferenceAnnotationMapper(
            answer1_key="response_a",
            answer2_key="response_b",
            prompt_key="question",
            chosen_key="chosen",
            rejected_key="rejected",
            wait_for_annotations=True
        )

        # Add mock annotation
        task_id = 3000
        mapper.add_mock_annotation(task_id, [
            {
                "type": "pairwise",
                "value": {
                    "selected": "left"  # Select response_a
                }
            }
        ])

        # Process the sample
        result = mapper.process_batched(custom_dict)

        # Verify the results
        self.assertIn("chosen", result)
        self.assertEqual(result["chosen"][0], result["response_a"][0])

    def test_process_uses_existing_ids(self):
        """Test that the Human Preference mapper uses existing IDs in samples instead of generating new ones"""
        # First pass: process without waiting for annotations
        mapper = MockHumanPreferenceAnnotationMapper(
            wait_for_annotations=False)

        # Create samples with predefined IDs
        samples_with_ids = {
            "prompt": ["Which is better? A or B", "Which is clearer? X or Y"],
            "answer1": ["Option A", "Option X"],
            "answer2": ["Option B", "Option Y"],
            "id": ["preference_id_1", "preference_id_2"]
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

        # Add mock annotations for the created tasks
        for i, task_id in enumerate(mapper.created_task_ids):
            # Alternate between left and right selections
            selected = "left" if i % 2 == 0 else "right"

            mapper.add_mock_annotation(task_id, [
                {
                    "type": "pairwise",
                    "value": {
                        "selected": selected
                    }
                }
            ])

        # Second pass: process with waiting for annotations
        mapper.wait_for_annotations = True
        result = mapper.process_batched(samples_with_ids)

        # Verify results include preference annotations
        self.assertIn("chosen", result)

        # First sample should prefer answer1 (left choice)
        self.assertEqual(result["chosen"][0], result["answer1"][0])

        # Second sample should prefer answer2 (right choice)
        self.assertEqual(result["chosen"][1], result["answer2"][1])

        # Verify the original IDs were preserved in the result
        self.assertEqual(result["id"], samples_with_ids["id"])


if __name__ == '__main__':
    unittest.main()
