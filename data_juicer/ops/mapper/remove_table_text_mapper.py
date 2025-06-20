import regex as re
from pydantic import Field
from typing_extensions import Annotated

from ..base_op import OPERATORS, Mapper


@OPERATORS.register_module("remove_table_text_mapper")
class RemoveTableTextMapper(Mapper):
    """
    Mapper to remove table texts from text samples.

    Regular expression is used to remove tables in the range of column
    number of tables.
    """

    _batched_op = True

    def __init__(
        self,
        min_col: Annotated[int, Field(ge=2, le=20)] = 2,
        max_col: Annotated[int, Field(ge=2, le=20)] = 20,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param min_col: The min number of columns of table to remove.
        :param max_col: The max number of columns of table to remove.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_col = min_col
        self.max_col = max_col
        self.pattern = r"(?<=\n)((\S+?)([ |\t](\S+?)){%d}\n+){2,}"

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            for i in range(self.min_col - 1, self.max_col):
                pattern = re.compile(self.pattern % i)
                text = pattern.sub("", text)

            samples[self.text_key][idx] = text

        return samples
