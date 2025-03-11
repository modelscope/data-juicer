import os
from typing import Dict, List, Optional

from loguru import logger

from data_juicer.ops.mapper.annotation.annotation_mapper import \
    LabelStudioAnnotationMapper

from ...base_op import OPERATORS


@OPERATORS.register_module('human_preference_annotation_mapper')
class HumanPreferenceAnnotationMapper(LabelStudioAnnotationMapper):
    """Operator for human preference annotation using Label Studio."""

    def __init__(self,
                 label_config_file: str = None,
                 answer1_key: str = 'answer1',
                 answer2_key: str = 'answer2',
                 text_key: str = 'prompt',
                 **kwargs):
        """Initialize the human preference annotation operator.

        Args:
            label_config_file: Path to the XML template file for Label Studio
            answer1_key: Field name for the first answer option
            answer2_key: Field name for the second answer option
            **kwargs: Additional arguments passed to LabelStudioAnnotationOp
        """
        # Load label config from file if provided
        if label_config_file and os.path.exists(label_config_file):
            with open(label_config_file, 'r') as f:
                kwargs['label_config'] = f.read()
                logger.info(f'Loaded label config from {label_config_file}')

        # Store data field keys
        self.answer1_key = answer1_key
        self.answer2_key = answer2_key
        self.text_key = text_key

        # Initialize the parent class
        super().__init__(**kwargs)

    def _format_task(self, samples: List[Dict]) -> Dict:
        """Format samples as a Label Studio task for human preference.

        Args:
            samples: List of samples to include in the task

        Returns:
            Dict: Formatted task data
        """
        # For human preference, we need a special format
        if len(samples) != 1:
            logger.warning(
                'Human preference requires exactly one sample per task')

        sample = samples[0]
        task = {'data': {}}

        # Add the prompt/question
        if self.text_key in sample:
            task['data']['prompt'] = sample[self.text_key]
        else:
            logger.warning(f"Sample missing required field '{self.text_key}'")
            task['data']['prompt'] = 'No prompt provided'

        # Add the answer options
        if self.answer1_key in sample:
            task['data']['answer1'] = sample[self.answer1_key]
        else:
            logger.warning(
                f"Sample missing required field '{self.answer1_key}'")
            task['data']['answer1'] = 'No answer 1 provided'

        if self.answer2_key in sample:
            task['data']['answer2'] = sample[self.answer2_key]
        else:
            logger.warning(
                f"Sample missing required field '{self.answer2_key}'")
            task['data']['answer2'] = 'No answer 2 provided'

        # Add any other metadata as string values only
        for key, value in sample.items():
            if key not in [self.text_key, self.answer1_key, self.answer2_key]:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    # Convert to string to ensure compatibility
                    task['data'][f'meta:{key}'] = str(
                        value) if value is not None else ''

        # Log the task for debugging
        logger.debug(f'Formatted task: {task}')

        return task

    def _get_task_annotation(self, task_id: int) -> Optional[Dict]:
        """Get annotation for a task if available"""
        annotation = super()._get_task_annotation(task_id)

        # Process the annotation if available
        if annotation and 'result' in annotation:
            # Extract the preference information
            for item in annotation['result']:
                if item.get('type') == 'pairwise':
                    # Get the selected option (from_id or to_id)
                    selected = item.get('value', {}).get('selected')
                    if selected:
                        # Add the preference to the annotation
                        annotation['preference'] = selected

        return annotation

    def process_dataset(self, dataset):
        """Process the dataset with custom field handling.

        This method is called by the executor to process the dataset.
        We override it to handle datasets with custom field names.
        """
        # Check if the dataset has the required fields
        required_fields = [self.text_key, self.answer1_key, self.answer2_key]
        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(
                    f"Required field '{field}' not found in dataset")

        # Call the parent class's process_dataset method
        return super().process_dataset(dataset)
