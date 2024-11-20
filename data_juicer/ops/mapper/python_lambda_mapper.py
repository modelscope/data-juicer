import ast

from ..base_op import OPERATORS, Mapper

OP_NAME = 'python_lambda_mapper'


@OPERATORS.register_module(OP_NAME)
class PythonLambdaMapper(Mapper):

    def __init__(self, lambda_str: str):
        # Parse and validate the lambda function
        self.lambda_func = self._create_lambda(lambda_str)

    def _create_lambda(self, lambda_str: str):
        # Parse input string into an AST and check for a valid lambda function
        try:
            node = ast.parse(lambda_str, mode='eval')

            # Check if the body of the expression is a lambda
            if not isinstance(node.body, ast.Lambda):
                raise ValueError(
                    'Input string must be a valid lambda function.')

            # Check that the lambda has exactly one argument
            if len(node.body.args.args) != 1:
                raise ValueError(
                    'Lambda function must have exactly one argument.')

            # Compile the AST to code
            compiled_code = compile(node, '<string>', 'eval')
            # Safely evaluate the compiled code allowing built-in functions
            func = eval(compiled_code, {'__builtins__': __builtins__})
            return func
        except Exception as e:
            raise ValueError(f'Invalid lambda function: {e}')

    def process_single(self, sample):
        # Process the input through the lambda function and return the result
        result = self.lambda_func(sample)

        # Check if the result is a valid
        if not isinstance(result, dict):
            raise ValueError(f'Lambda function must return a dictionary, '
                             f'got {type(result).__name__} instead.')

        return result
