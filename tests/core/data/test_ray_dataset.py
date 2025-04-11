import unittest
from data_juicer.utils.unittest_utils import TEST_TAG

class TestRayDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        self.data = [
            {
                'text': 'Hello',
                'score': 1,
                'metadata': {'lang': 'en'},
                'labels': [1, 2, 3]
            },
            {
                'text': 'World',
                'score': 2,
                'metadata': {'lang': 'es'},
                'labels': [4, 5, 6]
            },
            {
                'text': 'Test',
                'score': 3,
                'metadata': {'lang': 'fr'},
                'labels': [7, 8, 9]
            }
        ]

        # Create fresh dataset for each test
        self.dataset = RayDataset(ray.data.from_items(self.data))

    def tearDown(self):
        """Clean up test data"""
        self.dataset = None

    @TEST_TAG('ray')
    def test_get_column_basic(self):
        """Test basic column retrieval"""
        # Test string column
        texts = self.dataset.get_column('text')
        self.assertEqual(texts, ['Hello', 'World', 'Test'])

        # Test numeric column
        scores = self.dataset.get_column('score')
        self.assertEqual(scores, [1, 2, 3])

        # Test dict column
        metadata = self.dataset.get_column('metadata')
        self.assertEqual(metadata, [
            {'lang': 'en'},
            {'lang': 'es'},
            {'lang': 'fr'}
        ])

        # Test list column
        labels = self.dataset.get_column('labels')
        self.assertEqual(labels, [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

    @TEST_TAG('ray')
    def test_get_column_with_k(self):
        """Test column retrieval with k limit"""
        # Test k=2
        texts = self.dataset.get_column('text', k=2)
        self.assertEqual(texts, ['Hello', 'World'])

        # Test k larger than dataset
        texts = self.dataset.get_column('text', k=5)
        self.assertEqual(texts, ['Hello', 'World', 'Test'])

        # Test k=0
        texts = self.dataset.get_column('text', k=0)
        self.assertEqual(texts, [])

        # Test k=1
        texts = self.dataset.get_column('text', k=1)
        self.assertEqual(texts, ['Hello'])

    @TEST_TAG('ray')
    def test_get_column_errors(self):
        """Test error handling"""
        # Test non-existent column
        with self.assertRaises(KeyError) as context:
            self.dataset.get_column('nonexistent')
        self.assertIn("not found in dataset", str(context.exception))

        # Test negative k
        with self.assertRaises(ValueError) as context:
            self.dataset.get_column('text', k=-1)
        self.assertIn("must be non-negative", str(context.exception))

    @TEST_TAG('ray')
    def test_get_column_empty_dataset(self):
        """Test with empty dataset"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        empty_dataset = RayDataset(ray.data.from_items([]))

        # Should raise ValuError for empty dataset/columns
        with self.assertRaises(KeyError):
            empty_dataset.get_column('text')

    @TEST_TAG('ray')
    def test_get_column_types(self):
        """Test return type consistency"""
        # All elements should be strings
        texts = self.dataset.get_column('text')
        self.assertTrue(all(isinstance(x, str) for x in texts))

        # All elements should be ints
        scores = self.dataset.get_column('score')
        self.assertTrue(all(isinstance(x, int) for x in scores))

        # All elements should be dicts
        metadata = self.dataset.get_column('metadata')
        self.assertTrue(all(isinstance(x, dict) for x in metadata))

        # All elements should be lists
        labels = self.dataset.get_column('labels')
        self.assertTrue(all(isinstance(x, list) for x in labels))

    @TEST_TAG('ray')
    def test_get_column_preserve_order(self):
        """Test that column order is preserved"""
        texts = self.dataset.get_column('text')
        self.assertEqual(texts[0], 'Hello')
        self.assertEqual(texts[1], 'World')
        self.assertEqual(texts[2], 'Test')

        # Test with k
        texts = self.dataset.get_column('text', k=2)
        self.assertEqual(texts[0], 'Hello')
        self.assertEqual(texts[1], 'World')

    @TEST_TAG('ray')
    def test_schema_single_dataset(self):
        """Test schema for single dataset"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [{'text': 'hello', 'score': 1}]
        dataset = RayDataset(ray.data.from_items(data))

        schema = dataset.schema()
        self.assertIn('text', schema.columns)
        self.assertEqual(schema.column_types['text'], str)
        self.assertEqual(schema.column_types['score'], int)

    @TEST_TAG('ray')
    def test_schema_multiple_datasets(self):
        """Test schema consistency across multiple datasets"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data1 = [{'text': 'hello', 'score': 1}]
        data2 = [{'text': 'world', 'score': 2}]

        dataset1 = RayDataset(ray.data.from_items(data1))
        dataset2 = RayDataset(ray.data.from_items(data2))

        # Verify schemas match
        schema1 = dataset1.schema()
        schema2 = dataset2.schema()

        self.assertEqual(schema1.columns, schema2.columns)
        self.assertEqual(schema1.column_types, schema2.column_types)

    @TEST_TAG('ray')
    def test_schema_validation(self):
        """Test schema validation"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        # Test with invalid schema (mixed types)
        data = [
            {'text': 'hello', 'value': 1},
            {'text': 'world', 'value': 'string'}  # Mixed int/str
        ]

        with self.assertRaises(Exception) as context:
            dataset = RayDataset(ray.data.from_items(data))
            _ = dataset.schema()

        # Ray might choose either type
        self.assertIn("unable to merge", str(context.exception).lower())

    @TEST_TAG('ray')
    def test_schema_nested_structures(self):
        """Test schema with nested data structures"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [{
            'text': 'hello',
            'int_value': 1,
            'float_value': 1.0,
            'bool_value': True,
            'metadata': {'lang': 'en', 'score': 1},
            'tags': ['tag1', 'tag2'],
            'nested': {'a': {'b': {'c': 1}}}
        }]

        dataset = RayDataset(ray.data.from_items(data))
        schema = dataset.schema()

        self.assertEqual(schema.column_types['text'], str)
        self.assertEqual(schema.column_types['int_value'], int)
        self.assertEqual(schema.column_types['float_value'], float)
        self.assertEqual(schema.column_types['bool_value'], bool)
        self.assertEqual(schema.column_types['metadata'], dict)
        self.assertEqual(schema.column_types['tags'], list)
        self.assertEqual(schema.column_types['nested'], dict)

    @TEST_TAG('ray')
    def test_schema_empty_dataset(self):
        """Test schema with empty dataset"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset

        with self.assertRaises(ValueError) as context:
            dataset = RayDataset(ray.data.from_items([]))
            _ = dataset.schema()

        self.assertIn("empty", str(context.exception).lower())

    @TEST_TAG('ray')
    def test_schema_special_characters(self):
        """Test schema with special characters in column names"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [{
            'normal': 1,
            'with.dot': 2,
            'with-dash': 3,
            '_underscore': 4,
            'with space': 5
        }]

        dataset = RayDataset(ray.data.from_items(data))
        schema = dataset.schema()

        expected_columns = {
            'normal', 'with.dot', 'with-dash',
            '_underscore', 'with space'
        }
        self.assertEqual(set(schema.columns), expected_columns)

    @TEST_TAG('ray')
    def test_schema_type_consistency(self):
        """Test schema type consistency across rows"""
        import ray.data
        from data_juicer.core.data.ray_dataset import RayDataset
        data = [
            {'text': 'hello', 'score': 1, 'flag': True},
            {'text': 'world', 'score': 2, 'flag': False},
            {'text': 'test', 'score': 3, 'flag': True}
        ]

        dataset = RayDataset(ray.data.from_items(data))
        schema = dataset.schema()

        # Get all rows for type checking
        rows = list(dataset.data.take())

        # Verify types are consistent
        self.assertTrue(all(
            isinstance(row['text'], schema.column_types['text'])
            for row in rows
        ))
        self.assertTrue(all(
            isinstance(row['score'], schema.column_types['score'])
            for row in rows
        ))
        self.assertTrue(all(
            isinstance(row['flag'], schema.column_types['flag'])
            for row in rows
        ))

    @TEST_TAG('ray')
    def test_get(self):
        """Test get method for RayDataset"""
        import ray
        from data_juicer.core.data.ray_dataset import RayDataset

        # Test with simple data
        simple_data = [
            {'text': 'hello', 'score': 1},
            {'text': 'world', 'score': 2},
            {'text': 'test', 'score': 3}
        ]
        dataset = RayDataset(ray.data.from_items(simple_data))

        # Basic get
        rows = dataset.get(2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {'text': 'hello', 'score': 1})
        self.assertEqual(rows[1], {'text': 'world', 'score': 2})

        # Test with nested structures
        nested_data = [
            {
                'text': 'hello',
                'metadata': {'lang': 'en', 'source': 'web'},
                'tags': [1, 2, 3]
            },
            {
                'text': 'world',
                'metadata': {'lang': 'es', 'source': 'book'},
                'tags': [4, 5, 6]
            }
        ]
        nested_dataset = RayDataset(ray.data.from_items(nested_data))

        # Test nested structure preservation
        rows = nested_dataset.get(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['metadata']['lang'], 'en')
        self.assertEqual(rows[0]['tags'], [1, 2, 3])

        # Test edge cases
        self.assertEqual(dataset.get(0), [])
        self.assertEqual(len(dataset.get(10)), 3)  # More than dataset size
        with self.assertRaises(ValueError):
            dataset.get(-1)

        # Test type preservation
        row = dataset.get(1)[0]
        self.assertIsInstance(row, dict)
        self.assertIsInstance(row['text'], str)
        self.assertIsInstance(row['score'], int)


if __name__ == '__main__':
    unittest.main()