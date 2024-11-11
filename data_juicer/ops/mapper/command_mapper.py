import json
import subprocess

from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Mapper

json_repair = LazyLoader('json_repair', 'json_repair')

OP_NAME = 'command_mapper'


@OPERATORS.register_module(OP_NAME)
class CommandMapper(Mapper):
    """Mapper to execute inline code or an external script file.

    This class allows executing Python or shell scripts, taking input in
    JSON format and expecting JSON output.
    """

    def __init__(self, command: str = '', repair=True, **kwargs):
        """
        Initialization method.

        :param command: The command to execute. If an empty string is given,
            the input sample is return unchanged.
        :param repair: If True, uses `json_repair` to load JSON; otherwise,
            uses the standard `json` library.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.command = command
        self.repair = repair
        if self.repair:
            self.json_loads = json_repair.loads
        else:
            self.json_loads = json.loads

    def process_single(self, sample):
        if not self.command:
            return sample

        # Serialize the input data to JSON
        input_data = json.dumps(sample)

        # Execute the command, capturing both stdout and stderr
        result = subprocess.run(self.command,
                                input=input_data,
                                capture_output=True,
                                text=True,
                                shell=True)

        if result.returncode != 0:
            raise Exception(f'Execution failed: {result.stderr}')

        return self.json_loads(result.stdout)  # Return the parsed JSON output
