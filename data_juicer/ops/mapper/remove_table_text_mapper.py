import regex as re
from jsonargparse.typing import restricted_number_type

from ..base_op import OPERATORS, Mapper

from_2_to_20 = restricted_number_type('from_2_to_20', int, [('>=', 2),
                                                            ('<=', 20)])


@OPERATORS.register_module('remove_table_text_mapper')
class RemoveTableTextMapper(Mapper):
    """
    Mapper to remove table texts from text samples.

    Regular expression is used to remove tables in the range of column
    number of tables.
    """

    def __init__(self,
                 min_col: from_2_to_20 = 2,
                 max_col: from_2_to_20 = 20,
                 *args,
                 **kwargs):
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
        self.pattern = r'(?<=\n)((\S+?)([ |\t](\S+?)){%d}\n+){2,}'

    def process(self, sample):

        text = sample[self.text_key]
        for i in range(self.min_col - 1, self.max_col):
            pattern = re.compile(self.pattern % i)
            text = pattern.sub('', text)

        sample[self.text_key] = text
        return sample
