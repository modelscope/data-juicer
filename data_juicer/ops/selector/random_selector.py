from typing import Optional

from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.ops.base_op import OPERATORS, Selector
from data_juicer.utils.sample import random_sample


@OPERATORS.register_module("random_selector")
class RandomSelector(Selector):
    """Selector to random select samples."""

    def __init__(
        self,
        select_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
        select_num: Optional[PositiveInt] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param select_ratio: The ratio to select. When both
            select_ratio and select_num are set, the value corresponding
            to the smaller number of samples will be applied.
        :param select_num: The number of samples to select. When both
            select_ratio and select_num are set, the value corresponding
            to the smaller number of samples will be applied.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.select_ratio = select_ratio
        self.select_num = select_num

    def process(self, dataset):
        if len(dataset) <= 1:
            return dataset

        if self.select_ratio is None and self.select_num is None:
            return dataset

        if not self.select_ratio:
            select_num = self.select_num
        else:
            select_num = int(self.select_ratio * len(dataset))
            if self.select_num and self.select_num < select_num:
                select_num = self.select_num

        return random_sample(dataset, sample_number=select_num)
