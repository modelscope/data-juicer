import json
import unittest

from data_juicer.ops.mapper.command_mapper import CommandMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestCommandMapper(DataJuicerTestCaseBase):

    def test_execute_command(self):
        # Test that the command executes successfully and returns
        # the correct JSON output.
        sample = {}
        op = CommandMapper(command='echo \'{"name": "Alice", "age": 30}\'',
                           repair=False)
        result = op.process_single(sample)
        expected = {'name': 'Alice', 'age': 30}
        self.assertEqual(result, expected)

    def test_empty_command(self):
        # Test that an empty command returns the original input sample.
        op = CommandMapper(command='')
        sample = {'key': 'value'}
        result = op.process_single(sample)
        self.assertEqual(result, sample)

    def test_json_parsing_error_no_repair(self):
        # Test that a JSON parsing error is raised for invalid JSON output
        # with repair=False.
        faulty_command = 'echo \'{"name": "Alice", "age": 30\''
        op_no_repair = CommandMapper(command=faulty_command, repair=False)
        sample = {}
        with self.assertRaises(json.JSONDecodeError):
            op_no_repair.process_single(sample)

    def test_json_parsing_error_with_repair(self):
        # Test that the command output can be repaired and
        # correctly parsed when repair=True.
        faulty_command = 'echo \'{"name": "Alice", "age": 30\''
        op_with_repair = CommandMapper(command=faulty_command, repair=True)
        sample = {}
        result = op_with_repair.process_single(sample)
        expected = {'name': 'Alice', 'age': 30}
        self.assertEqual(result, expected)

    def test_command_execution_failure(self):
        # Test that an exception is raised when the command execution fails.
        faulty_command = 'ls non_existent_file'
        op = CommandMapper(command=faulty_command)
        sample = {}
        with self.assertRaises(Exception) as context:
            op.process_single(sample)
        self.assertTrue('Execution failed:' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
