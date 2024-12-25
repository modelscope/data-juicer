from unittest import TestCase, main
import datasets
import ray
import ray.data
import pandas as pd

from data_juicer.core.data import NestedDataset, RayDataset
from data_juicer.core.data.data_validator import (DataValidationError, 
    RequiredFieldsValidator
)


# Test RequiredFieldsValidator
class TestRequiredFieldsValidator(TestCase):
    
    def setUp(self):
        # Create sample DataFrame
        self.df  = pd.DataFrame({
            'text': ['Hello', 'World', None, 'Test'],
            'metadata': [{'lang': 'en'}, {'lang': 'es'}, {'lang': 'fr'}, None],
            'score': [1.0, 2.0, 3.0, 4.0]
        })

        # Create dataset
        self.dataset = NestedDataset(datasets.Dataset.from_pandas(self.df))
        
        # Create ray dataset
        self.ray_dataset = RayDataset(ray.data.from_pandas(self.df))    


    def test_basic_validation(self):
        """Test basic field validation"""
        config = {
            'required_fields': ['text', 'metadata'],
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        
        # Should pass
        validator.validate(self.dataset)
        
        # Should fail with missing field
        config['required_fields'].append('nonexistent')
        validator = RequiredFieldsValidator(config)
        with self.assertRaises(DataValidationError) as exc:
            validator.validate(self.dataset)
        self.assertIn("missing required fields", str(exc.exception).lower())

    def test_type_validation(self):
        """Test field type validation"""
        # Should pass        
        config = {
            'required_fields': ['text', 'score'],
            'field_types': {
                'text': str,
                'score': float
            },
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        validator.validate(self.dataset)

        # Should fail with wrong type
        config['field_types']['score'] = str
        validator = RequiredFieldsValidator(config)
        with self.assertRaises(DataValidationError) as exc:
            validator.validate(self.dataset)
        self.assertIn("incorrect type", str(exc.exception).lower())

    def test_ray_dataset_support(self):
        """Test validation with RayDataset"""
        config = {
            'required_fields': ['text', 'metadata'],
            'field_types': {
                'text': str,
                'metadata': dict
            },
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        
        # Should pass
        validator.validate(self.ray_dataset)

    def test_invalid_dataset_type(self):
        """Test validation with unsupported dataset type"""
        config = {
            'required_fields': ['text']
        }
        validator = RequiredFieldsValidator(config)
        
        with self.assertRaises(DataValidationError) as exc:
            validator.validate([1, 2, 3])  # Invalid dataset type
        self.assertIn("unsupported dataset type", str(exc.exception).lower())

    def test_empty_required_fields(self):
        """Test validation with empty required fields"""
        config = {
            'required_fields': []
        }
        validator = RequiredFieldsValidator(config)
        
        # Should pass as no fields are required
        validator.validate(self.dataset)

    def test_multiple_dataset_types(self):
        """Test validation works with different dataset types"""
        datasets_to_test = [
            ('nested', self.dataset),
            ('ray', self.ray_dataset)
        ]
        
        config = {
            'required_fields': ['text', 'metadata', 'score'],
            'allow_missing': .25
        }
        validator = RequiredFieldsValidator(config)
        
        for name, dataset in datasets_to_test:
            with self.subTest(dataset_type=name):
                validator.validate(dataset)

if __name__ == '__main__':
    main()